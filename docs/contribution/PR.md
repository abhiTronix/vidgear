<!--
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019-2020 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
===============================================
-->

# Submitting Pull Request(PR) Guidelines:


The following guidelines tells you how to submit a valid PR for vidGear:

!!! question "Working on your first Pull Request for VidGear?" 

    * If you don't know how to contribute to an Open Source Project on GitHub, then You can learn about it from [here](https://opensource.guide/how-to-contribute/).

    * If you're stuck at something, please join our [Gitter community channel](https://gitter.im/vidgear/community). We will help you get started!

    * Kindly follow the [EXEMPLARY :medal:](https://github.com/abhiTronix/vidgear/pulls?q=is%3Apr+label%3A%22EXEMPLARY+%3Amedal_military%3A%22+) tag for some finest PR examples.


&nbsp; 

## Forking and Cloning

??? tip "First fork on GitHub?" 

    You can easily learn about it from [Fork a repo](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) wiki.


!!! danger "Pull Request Requirements"
    
    **Any Pull Request failed to comply following requirements will be rejected:**

    * The [`testing`](https://github.com/abhiTronix/vidgear/tree/testing) branch of your Forked repository **MUST** be up-to-date with VidGear, before starting working on Pull Request.
    * Your new working branch for Pull Request **MUST** be a sub-branch of [`testing`](https://github.com/abhiTronix/vidgear/tree/testing) branch of your Forked repository only.
    * All Pull Requests **MUST** be pushed against VidGear's [`testing`](https://github.com/abhiTronix/vidgear/tree/testing) branch only.



You can clone your forked remote git to local, and create your PR working branch as a sub-branch of `testing` branch as follows:


```sh
# clone your forked repository(change with your username) and get inside
git clone https://github.com/{YOUR USERNAME}/vidgear.git && cd vidgear

# checkout the latest testing branch
git checkout testing

# Now create your new branch with suitable name(such as "subbranch_of_testing")
git checkout -b subbranch_of_testing
```

Now after working with this newly created branch for your Pull Request, you can commit and push or merge it locally or remotely as usual.


&nbsp; 

## Submission Checklist

There are some important checks you need to perform while submitting your Pull Request(s) for VidGear library:

- [x] **Submit an issue and Link your Pull Request:**

    !!! tip "For more information on Linking a pull request to an issue, See [this wiki](https://docs.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue) doc"
  
  * If you would like to implement a new feature/improvement for VidGear, please submit an issue with a [proposal template](https://github.com/abhiTronix/vidgear/issues/new?labels=issue%3A+proposal&template=proposal.md) for your work first and then submit your Pull Request. 
  * You can link an issue to a pull request manually or using a supported keyword in the pull request description.
  * When you link a pull request to the issue the pull request addresses, collaborators can see that someone is working on the issue. 

- [x] **Perform PR Integrity Checks:** 
    
  * Search GitHub for an open or closed PR that relates to your submission. Duplicate contributions will be rejected.
  * Submit the pull request from the first day of your development and create it as a draft pull request. Click ready for review when finished and passed all the checks.
  * Check if your purposed code matches the overall direction, simplicity and structure of the VidGear APIs and improves it.
  * Make sure your PR must pass through all unit tests including VidGear's [CI tests](#testing-formatting-linting). If it's somehow failing, then ask maintainer for a review.
  * It is important to state that you retain copyright for your contributions, but also agree to license them for usage by the project and author(s) under the [Apache license](https://github.com/abhiTronix/vidgear/blob/master/LICENSE).

- [x] **Test, Format & lint locally:**

  * Make sure to locally test, format and lint the modified code every time you commit. The details are discussed above.

- [x] **Make sensible commit messages:**

  * If your PR fixes a separate issue number, remember to include `"resolves #issue_number"` in the commit message. Learn more about it [here](https://help.github.com/articles/closing-issues-using-keywords/).
  * Keep commit message concise as much as possible at every submit. You can make a supplement to the previous commit with `git commit --amend` command.
  * If we suggest changes, make the required updates, rebase your branch and push the changes to your GitHub repository, which will automatically update your PR.

- [x] **Draft the PR according to template:**

  * Remember to completely fill the whole template for PR. Incomplete ones will be subjected to re-edit!
  * Add a brief but descriptive title for your PR.
  * Explain what the PR adds, fixes or improves.
  * In case of bug fixes, add new unit test case which would fail against your bug fix.
  * Provide CLI commands and output or screenshots where you can.

&nbsp; 

## Testing, Formatting & Linting

All Pull Request(s) must be tested, formatted & linted against our library standards as discussed below:

### Requirements

Testing VidGear requires additional test dependencies and dataset, which can be handled manually as follows:

* **Install additional python libraries:**
  
    You can easily install these dependencies via pip:

    ```sh
    pip install --upgrade six, flake8, black, pytest, pytest-asyncio
    ```

* **Download Tests Dataset:** 

    To perform tests, you also need to download additional dataset *(to your temp dir)* by running [`prepare_dataset.sh`](https://github.com/abhiTronix/vidgear/blob/master/scripts/bash/prepare_dataset.sh)  bash script as follows:

    ```sh
    chmod +x scripts/bash/prepare_dataset.sh
    # On linux and MacOS
    .scripts/bash/prepare_dataset.sh
    # On Windows 
    sh scripts/bash/prepare_dataset.sh
    ```

### Running Tests

All tests can be run with [`pytest`](https://docs.pytest.org/en/stable/)(*in VidGear's root folder*) as follows:

   ```sh
    pytest -sv  #-sv for verbose output.
   ```

### Formatting & Linting

For formatting and linting, following libraries are used:

* **Flake8:** You must run [`flake8`](https://flake8.pycqa.org/en/latest/manpage.html) linting for checking the code base against the coding style (PEP8), programming errors and other cyclomatic complexity:

    ```sh
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    ```

* **Black:**  Vidgear follows [`black`](https://github.com/psf/black) formatting to make code review faster by producing the smallest diffs possible. You must run it with sensible defaults as follows: 

    ```sh
    black {source_file_or_directory}
    ```

&nbsp; 


## Frequently Asked Questions


**Q1. Why do my changes taking so long to be Reviewed and/or Merged?**

!!! info "Submission Aftermaths"
    * After your PR is merged, you can safely delete your branch and pull the changes from the main (upstream) repository.
    * The changes will remain in `testing` branch until next VidGear version is released, then it will be merged into `master` branch.
    * After a successful Merge, your newer contributions will be given priority over others. 

Pull requests will be reviewed by the maintainers and the rationale behind the maintainer’s decision to accept or deny the changes will be posted in the pull request. Please wait for our code review and approval, possibly enhancing your change on request.


**Q2. What if I want to submit my Work that is Still In Progress?**

You can do it. But please use one of these two prefixes to let reviewers know about the state of your work:

*  **[WIP]** _(Work in Progress)_: is used when you are not yet finished with your pull request, but you would like it to be reviewed. The pull request won't be merged until you say it is ready.
*  **[WCM]** _(Waiting Code Merge)_: is used when you're documenting a new feature or change that hasn't been accepted yet into the core code. The pull request will not be merged until it is merged in the core code _(or closed if the change is rejected)_.


**Q3. Would you accept a huge Pull Request with Lots of Changes?**

First, make sure that the changes are somewhat related. Otherwise, please create separate pull requests. Anyway, before submitting a huge change, it's probably a good idea to [open an issue](../../contribution/issue) in the VidGear Github repository to ask the maintainers if they agree with your proposed changes. Otherwise, they could refuse your proposal after you put all that hard work into making the changes. We definitely don't want you to waste your time!

&nbsp; 