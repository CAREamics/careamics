# Composition over inheritance

The principle of composition over inheritance is that, instead of using inheritance to extend or change the behaviour of a class, it can be composed of modules that can be swapped to extend or change behaviour.

The following basic example shows how inheritance can create hard to maintain code:

```python
# imports

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import tifffile
```

Say you want to have a set of image readers for different file types; you might think to create an abstract base class and implement subclasses for each file type.

```python
class ImageReader(ABC):

    @abstractmethod
    def read(self, path: Path) -> NDArray[Any]:
        ...

class NumpyImageReader(ImageReader):

    def read(self, path: Path) -> NDArray[Any]:
        img = np.load(path)
        return img
    
class TiffImageReader(ImageReader):

    def read(self, path: Path) -> NDArray[Any]:
        img = tifffile.imread(path)
        return img
```

This is working well, but then someone requests a feature for the images to be automatically normalised between 0 and 1. So you create a second abstract class `NormImageReader` that inherits from the original `ImageReader` abstract class.

Now to combine the concrete reading functionality with the normalisation functionality,
additional subclasses have to be created.

```python
class NormImageReader(ImageReader, ABC):

    def norm(self, img: NDArray[Any]) -> NDArray[Any]:
        vmin = img.min()
        vmax = img.max()
        return (img - vmin)/(vmax - vmin)

class NormNumpyImageReader(NumpyImageReader, NormImageReader):

    def read(self, path: Path) -> NDArray[Any]:
        img = super().read(path)
        normed_img = super().norm(img)
        return normed_img
    
class NormTiffImageReader(TiffImageReader, NormImageReader):

    def read(self, path: Path) -> NDArray[Any]:
        img = super().read(path)
        normed_img = super().norm(img)
        return normed_img
```

But I'm sure the problem is now clear that if you needed to add another file type reader you have to add two additional concrete classes. And if you needed to add a different normalisation method then you would have to add a new concrete class for each file type. Quickly there is a subclass explosion. It also means the normalisation function is not easily reusable because it is bound to the `NormImageReader` class.

Clearly this is a contrived example, but the key message is, inheritance might seem like a simple solution at the start of a project, but additional unforseen features can mean it complexifies the code.

The solution is composition. If the `ImageReader` was designed to be a single class that could be injected with a read function, then the addition of a `norm_func` at a later date that could also be injected would be a simple feature addition. The following code shows how the example could be refactored to use composition.

```python
ReadFunc = Callable[[Path], NDArray[Any]]
NormFunc = Callable[[NDArray[Any]], NDArray[Any]]

def identity_norm(img: NDArray[Any]) -> NDArray[Any]:
    return img

class ImageReader:

    def __init__(self, read_func: ReadFunc, norm_func: Optional[NormFunc]):
        self.read_func: ReadFunc = read_func
        self.norm_func: NormFunc = norm_func if norm_func else identity_norm

    def read(self, path: Path) -> NDArray[Any]:
        img = self.read_func(path)
        if self.norm_func is not None:
            normed_img = self.norm_func(img)
        else:
            normed_img = img
        return normed_img
    
def min_max_norm(img: NDArray[Any]) -> NDArray[Any]:
    vmin = img.min()
    vmax = img.max()
    return (img - vmin)/(vmax - vmin)

def zero_mean_unit_std_norm(img: NDArray[Any]) -> NDArray[Any]:
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean)/std

reader = ImageReader(read_func=np.load, norm_func=zero_mean_unit_std_norm)
```