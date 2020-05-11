<a name="top"></a>

# Premier Converter: Example Data
Notes about the example data files for use in the project.

<!--This table of contents is maintained *manually*-->
## Contents
1. [Important information](#Important-information)
1. [Example files](#Example-files)
1. [Further notes](#Further-notes)

<p align="right"><a href="#top">Back to top</a></p>

## Important information
The data files used in this repo has been randomised and all information has been masked so they can be used for training purposes.

<p align="right"><a href="#top">Back to top</a></p>

## Example files
### `minimal01_dummy_data.xlsx`
Example of a small input data set (sheet `Sheet1`) which should not throw any errors, along with the expected output (sheet `expected_result`).

Specific functionality included:
- Multiple perils (2)
- Multiple factors (3)
- Not every row has every peril-factor combination
- Variable number of factor sets, various orders, blank factor set.
- Multiple sheets in the input workbook. Input data in the first sheet.
- Consistent values for the value fields.
- Relativity that is so close to 1 that the incremental premium is zero.
- Example of an error row (1)
- Unwanted data in columns of each row, starting with a cell with value `Total Peril Premium`. Occurs on every row except the error row.

<p align="right"><a href="#top">Back to top</a></p>

## Further notes
- Excel 'trims' excess columns and rows when a workbook is saved, so we'll assume that there will be values in the right-most column and lowest row (and we don't need to test for it). There could be a column with no value, but with formatting (or included for some other reason). This has not been tested.

<p align="right"><a href="#top">Back to top</a></p>
