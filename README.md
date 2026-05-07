# ByzKit

ByzKit is a toolkit for dataset synthesis and machine learning model training for byzantine chant notation OMR.

The project is a WIP and the README serves rather as a roadmap than a documentation of currently implemented functionalities.

## The Byz

This project is inspired by the [Neanes](https://github.com/neanes/neanes) and its complementary projects.

The original idea was to use Neanes to generate dataset that could be later augmented in order to train a model.
Due to the design of Neanes application (which is a WYSIWYG editor for sheet score-writting)
full automatization of synthesis process is very diffucult if not nearly impossible and it requires a graphical runtime.

Therefore I took the Neanes font (`byztex/Neanes.otf`),
which follows the SBMuFL standard, proposed by the Neanes team, and I created few basic LuaLatex templates
that are to be used for dataset synthesis.
So far I only use `byztex/template_standalone.tex`, the other templates are WIP.

## The Kit

The toolkit consits of 3 main programs that essentially form a toolchain:

1. dataset synthetiser
2. model training script
3. interactive demo

### Dataset synthetiser

The dataset synthetiser works in 2 main operations modes.

- raw dataset synthesis
- binary dataset synthesis

The raw dataset is to be merely an intermediate product of the synthesis process
where the program only generates a given number of images &mdash; ordered PNG files &mdash;
alongside a single text file containing all the labels in corresponding order.
Such data shall be considered the golden data.

The binary dataset is then a dataset in a form ready to be used by the model training script including:

- augmentation
- split (on the golden data)
- data-target paired form
