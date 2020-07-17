# Custom PyTorch Dataset class

## Description

This repo implements several custom PyTorch dataset classes for image datasets that use Numpy file formats(`.npy`, `.npz`). It integrates it with a standard CNN image classifier.

The classes implemented so far:

* `MemapDataset`: leverages Numpy's memory-mapped arrays. This can speed up training by allowing access 
to small fragments of large files without reading the entire file into memory.

* `MultiNpzDataset`: assumes an image dataset that is split over many `.npz` files. The objective of this class 
is to keep the advantages of compression while still having a fast way to load images during training.

## Use-cases

To enjoy the benefits of the respective classes, image data must be in some specific formats:
* `MemapDataset`
    * Provide a `.lst` file referencing your images with their classes, with the following format:
    ```bash
    img_idx \t img_label \t img_path  
    ```  
    * Store your images as `.npy` generated from a memmap array, for instance:
    ```python
    x = np.memmap('/images.npy', mode='w+', dtype=np.ubyte, 
              shape=(int(1E10),), offset=128)
    ```
    
* `MultiNpzDataset` images must be arranged as follows:
    * Split your images in groups (here by equal groups but not necessarily). The size here is very flexible (from one image to potentially thousands), 
    and should be chosen depending on the context, as this greatly impacts the loading time at training.
    ``` 
     data-01-1000.npz, data-1001-2000.npz, ...
    ```
  
    * The first file would be obtained as follow:
    ```python
    np.savez_compressed(
        "data-01-1000.npy",
        {
            "train" : stacked_training_images_arrays,
            "labels": stacked_corresponding_training_labels_arrays
        }
    )
    ```

