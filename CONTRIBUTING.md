# Contributing to Senti-Bot

Thank you for taking interest in contributing to the sentimental analysis bot.<br>
The following are guidelines for contributing to the project.


## License

License is [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).<br>
By contributing to the project you agree to abide by it.


## Getting Started

- Clone the repository and make sure it runs with `python3 main.py` (may require creating a .env file with valid token).
- Branch out from `master`.
- Please keep commits small and passing existing tests.
- Any new kernels created should extend the class `kernels.kernel.AbstractKernel`.
- Create applicable tests for new methods as needed. All non-bot methods (so anything outside of `main.py`) should be tested. Discord bot related commands are difficult to test hence are not mandatory to be unit tested _but_ should be tested manually by code-author.
- Create a pull request citing issue that is fixed. All code must be up to date with the `master` branch at the time of submitting the request.


## Code of Conduct

Keep all commits and conversation civil and sans any explicit language.<br>
For any queries or if stuck with anything, contact Nikolai#4151 in the [CSCH Discord](https://discord.gg/ndFR4RF).