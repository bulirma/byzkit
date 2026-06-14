# ByzKit

ByzKit is a toolkit for dataset synthesis and
machine learning model training for byzantine chant notation OMR.

The project is a WIP and the README serves rather as a roadmap
than a documentation of currently implemented functionalities.

## The Byz

This project is inspired by the [Neanes](https://github.com/neanes/neanes) project and
its complementary projects.

The original idea was to use Neanes to generate dataset that could be later augmented
in order to train a model.
Due to the design of Neanes application (which is a WYSIWYG editor for sheet score-writting)
full automatization of the synthesis process is very diffucult if not nearly impossible and
it requires a graphical runtime.
Besides, when generating the data automatically, Neanes incorrectly handles
the typesetting of large subset of neumes.

Therefore I took the fonts (in `byztex/fonts`) used by Neanes,
which follow the SBMuFL standard, proposed by the Neanes team, and
I created few basic LuaLatex templates
that are to be used for dataset synthesis.

## The Kit

The toolkit consits of 4 tools:

1. [dataset synthetiser](docs/dataset_synthetiser.md)
2. model trainer
3. demo tool(s)
4. BYZX converter

for more information and **usage** read the documentation linked in the list above.

## Requirements

- LuaLaTeX
- [fonts](byztex/fonts) installed
- python 3.11+
- python packages in [requirements](requirements.txt)
