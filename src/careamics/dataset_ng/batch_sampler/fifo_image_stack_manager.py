from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..patch_extractor.image_stack import ManagedLazyImageStack


class FifoImageStackManager:

    def __init__(self, max_files_loaded: int = 1):
        self.image_stacks: dict[Path, ManagedLazyImageStack] = {}
        self.loaded_image_stacks: list[ManagedLazyImageStack] = []
        self.max_files_loaded = max_files_loaded

    @property
    def currently_loaded(self):
        return len(self.loaded_image_stacks)

    def register_image_stack(self, image_stack: "ManagedLazyImageStack"):
        self.image_stacks[image_stack.path] = image_stack
        image_stack.set_callbacks(
            on_load=self.notify_load,
            on_close=self.notify_close,
        )

    def register_image_stacks(self, image_stacks: Sequence["ManagedLazyImageStack"]):
        for image_stack in image_stacks:
            self.register_image_stack(image_stack)

    def free(self):
        if self.currently_loaded >= self.max_files_loaded:
            image_stack_to_close = self.loaded_image_stacks[0]  # FIFO
            image_stack_to_close.deallocate()

    def notify_load(self, image_stack: "ManagedLazyImageStack"):
        if image_stack.path not in self.image_stacks:
            raise ValueError(
                f"Image stack with path {image_stack.path} has not been registered."
            )
        # TODO: Raise error if already loaded?
        self.free()
        self.loaded_image_stacks.append(image_stack)

    def notify_close(self, image_stack: "ManagedLazyImageStack"):
        if image_stack.path not in self.image_stacks:
            raise ValueError(
                f"Image stack with path {image_stack.path} has not been registered."
            )
        loaded_paths = [img_stack.path for img_stack in self.loaded_image_stacks]
        file_id = loaded_paths.index(image_stack.path)  # TODO: this will be file path
        self.loaded_image_stacks.pop(file_id)

    def close_all(self):
        loaded_image_stacks = self.loaded_image_stacks.copy()
        for image_stack in loaded_image_stacks:
            image_stack.deallocate()
