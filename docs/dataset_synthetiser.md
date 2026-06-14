# Dataset synthetiser

Dataset synthetiser can generate 4 types of dataset:

1. *page* dataset
2. *line* dataset
3. *default* (or *db*) dataset
4. *split* (or *sdb*) dataset

Each dataset type can be used to create dataset of any higher type
(it has greater number in the type).
All of the types have its purpose
however only the last one can be used for training with ByzKit.

> [!Note]
> LMDB (at least its python bindings) use different terminology than relational databases.
> The database as a whole is called **_environment_** and then it can
> &mdash; optionaly, since it is not a relational database &mdash; contain **_databases_**
> (which would be the equivalent of tables).

## *Page* dataset

The *page* type is used to generate a dataset with LuaLaTeX.
The other types do not depend on LuaLaTeX and fonts being installed (nor other ByzKit tools).
Hence *page* dataset can be generated on a system with the dependencies
and other types can be generated on a system without.
That is useful, because the *line* dataset takes a lot of time (due to augmentations).
Thus it can be generated on a cluster which might not have those dependencies.

### Structure

- metadata file (JSON)
- directory for each page
    - image of the page (PNG)
    - text file containing labels (SBMuFL) in order (left-right, top-down)

## *Line* dataset

The *line* dataset still consits of images.
So it is easy to explore, though the labels are serialized in NPZ files.
In this dataset the samples are already paired (data and target).

The image samples are of 2 kinds:

- *raw*: the image of a line was segmented from the original page image
- *augmented*: the image of a line was segmented from augmented image of a page

### Structure

- metadata file (JSON)
- directory for *raw* data
    - directory for each page
        - images of lines (PNG)
        - mathing (by the name) NPZ files
- directory for *augmented* data
    - directory for each page
        - augmented images of lines (PNG)
        - mathing (by the name) NPZ files

## *Default* dataset

The *default* dataset is stored in LMDB format.
Due to its speed it is inteded for distribution
(the particular split of the data can be performed later, because it is really fast).

### Structure

- metadata file (JSON)
- binary lock file
- binary database file

The database *environment* contains 4 *databases*:

- *raw* data
- *raw* targets
- *augmented* data
- *augmented* targets

The keys are padded ~~strings~~ bytes containing index which is the number of the line in order.

## *Split* dataset

The *split* dataset is stored in LMDB format as well.
It is however already splitted and thus ready to be used for training.

### Structure

File structure is the same.
The way the data are stored is similar only with the difference that there are 6 *databases*
for training, validation and testing
(each of those has one *database* for data &mdash; images &mdash; and target).

## Usage

```bash
python3 byzkit.py dataset [arg]...
```

Use `--help` argument for more information.
