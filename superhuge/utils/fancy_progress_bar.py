import inspect
from collections.abc import Iterable
from typing import cast, Optional, Dict, Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import (
    CustomProgress,
    MetricsTextColumn,
)
from rich import get_console, reconfigure
from rich.console import RenderableType
from rich.progress import Task, TaskID
from rich.style import Style
from rich.table import Table
from rich.text import Text


class MetricsTableColumn(MetricsTextColumn):
    max_refresh = 5
    """A column containing table."""

    def __init__(
        self,
        trainer: pl.Trainer,
        style: str | Style,
        text_delimiter: str,
        metrics_format: str,
    ):
        super().__init__(
            trainer=trainer,
            style=style,
            text_delimiter=text_delimiter,
            metrics_format=metrics_format,
        )

    def render(self, task: "Task"):  # type: ignore
        """
        渲染进度条的方法。
        加了全局 try-except 保护，防止在训练结束清理阶段（teardown）因缓存清空导致的崩溃。
        """
        try:
            # 1. 基础类型检查
            if not isinstance(self._trainer.progress_bar_callback, RichProgressBar):
                return Text()

            # 2. 状态检查：非 fit 阶段、sanity check 或非主进度条时不渲染
            # 使用 getattr 防止部分属性在 teardown 时不可用
            train_bar_id = getattr(self._trainer.progress_bar_callback, 'train_progress_bar_id', None)
            
            # 安全获取状态
            state_fn = getattr(self._trainer.state, 'fn', None)
            sanity_checking = getattr(self._trainer, 'sanity_checking', False)

            if (
                state_fn != "fit"
                or sanity_checking
                or train_bar_id != task.id
            ):
                return Text()
            
            # 3. 任务切换逻辑与缓存安全访问
            # 使用 getattr 安全获取 _tasks 字典
            tasks_cache = getattr(self, "_tasks", {})
            render_cache = getattr(self, "_renderable_cache", {})
            
            if self._trainer.training and task.id not in tasks_cache:
                tasks_cache[task.id] = "None"
                
                # [关键修复] 安全访问 _renderable_cache
                # 在程序退出时，缓存可能已被清空，必须先检查 key 是否存在
                current_id = getattr(self, "_current_task_id", None)
                
                if render_cache and current_id is not None:
                    current_id = cast(TaskID, current_id)
                    # 只有当 ID 确实存在于缓存中时才进行赋值
                    if current_id in render_cache:
                        try:
                            # 尝试获取缓存内容，如果是 tuple 则取第二个元素（Rich 的通常结构）
                            content = render_cache[current_id]
                            if isinstance(content, (list, tuple)) and len(content) > 1:
                                tasks_cache[current_id] = content[1]
                            else:
                                tasks_cache[current_id] = content
                        except Exception:
                            pass # 忽略缓存读取错误
                
                self._current_task_id = task.id
                
            # 4. 返回非当前任务的静态内容
            if self._trainer.training and task.id != self._current_task_id:
                return tasks_cache.get(task.id, Text(""))

            # 5. 生成当前任务的动态表格
            return self._generate_metrics_table()

        except Exception:
            # [终极安全网]
            # 如果在渲染过程中发生任何错误（如 KeyError, AttributeError, IndexError 等），
            # 尤其是在 teardown 阶段，直接返回空文本，防止程序崩溃。
            return Text()

    def _generate_metrics_table(self):
        # 安全移除 v_num
        self._metrics.pop("v_num", None)
        
        train_metrics = {
            k.split("/", 1)[-1]: v
            for k, v in self._metrics.items()
            if isinstance(k, str) and "train" in k
        }
        val_metrics = {
            k.split("/", 1)[-1]: v
            for k, v in self._metrics.items()
            if isinstance(k, str) and "val" in k
        }
        test_metrics = {
            k.split("/", 1)[-1]: v
            for k, v in self._metrics.items()
            if isinstance(k, str) and "test" in k
        }

        all_keys = sorted(
            set(
                [
                    key.split("/", 1)[-1]
                    for key in self._metrics.keys()
                    if isinstance(key, str)
                ]
            )
        )

        # Construct a table
        table = Table(
            title="Training Metrics", show_header=True, header_style="bold magenta"
        )
        table.add_column("Metric", style="bold cyan", justify="left")
        table.add_column("Train", style="green", justify="right")
        table.add_column("Validation", style="yellow", justify="right")
        table.add_column("Test", style="red", justify="right")

        for key in all_keys:
            train_val = (
                f"{train_metrics.get(key, 'N/A'):.4f}" if key in train_metrics else "—"
            )
            val_val = (
                f"{val_metrics.get(key, 'N/A'):.4f}" if key in val_metrics else "—"
            )
            test_val = (
                f"{test_metrics.get(key, 'N/A'):.4f}" if key in test_metrics else "—"
            )
            table.add_row(key, train_val, val_val, test_val)
        return table


class MetricNextLineProgress(CustomProgress):

    def __init__(self, metric_table: MetricsTableColumn, *columns, **kwargs):
        self._metric_table = metric_table
        super().__init__(*columns, **kwargs)

    def get_renderables(self) -> Iterable[RenderableType]:
        """Get a number of renderables for the progress display."""
        yield self.make_tasks_table(self.tasks)
        yield self.make_metric_tasks_table(self.tasks)

    def make_metric_tasks_table(self, tasks: Iterable[Task]) -> Table:
        table_columns = self._metric_table.get_table_column().copy()
        table = Table.grid(table_columns, padding=(0, 1), expand=self.expand)

        for task in tasks:
            if task.visible:
                table.add_row(self._metric_table(task))
        return table


class FancyProgressBar(RichProgressBar):
    def __init__(self, refresh_rate: int = 5):
        parent_signature = inspect.signature(super().__init__)

        # Validate the arguments against the parent's signature
        bound_arguments = parent_signature.bind(refresh_rate=refresh_rate)
        bound_arguments.apply_defaults()  # Ensure default values are included

        # Forward the validated arguments to the parent
        super().__init__(*bound_arguments.args, **bound_arguments.kwargs)

    def _init_progress(self, trainer: "pl.Trainer") -> None:
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            self._console = get_console()
            
            # [关键修复] 移除了 clear_live()，防止 IndexError
            
            self._metric_component = MetricsTableColumn(
                trainer,
                self.theme.metrics,
                self.theme.metrics_text_delimiter,
                self.theme.metrics_format,
            )
            self.progress = MetricNextLineProgress(
                self._metric_component,
                *self.configure_columns(trainer),
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,  # type: ignore
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._update(self.train_progress_bar_id, batch_idx + 1)
        self._update_metrics(trainer, pl_module)
        if (batch_idx + 1) // self.refresh_rate == 0:
            self.refresh()

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        if (batch_idx + 1) // self.refresh_rate == 0:
            return super().on_validation_batch_start(
                trainer, pl_module, batch, batch_idx, dataloader_idx
            )