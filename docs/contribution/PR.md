<!--
===============================================
vidgear library source-code is deployed under the Apache 2.0 License:

Copyright (c) 2019 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

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

    * If you don't know how to contribute to an Open Source Project on GitHub, then You can learn about it from [here ➶](https://opensource.guide/how-to-contribute/)

    * If you're stuck at something, please join our [Gitter community channel](https://gitter.im/vidgear/community). We will help you get started!

    * Kindly follow the [EXEMPLARY :medal:](https://github.com/abhiTronix/vidgear/pulls?q=is%3Apr+label%3A%22EXEMPLARY+%3Amedal_military%3A%22+) tag for some finest PR examples.


&nbsp; 

## Clone Testing branch

!!! warning "Base Branch must be `testing` in your Pull Request"
	Every PR **MUST** be pushed against VidGear's [`testing`](https://github.com/abhiTronix/vidgear/tree/testing) branch only, in order to trigger must needed CI testing workflows. If your's not, then [change the base branch to `testing` ➶](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-base-branch-of-a-pull-request)

!!! tip "Make sure the [`testing`](https://github.com/abhiTronix/vidgear/tree/testing) branch of your Forked repository is up-to-date with VidGear, before starting working on Pull Request."

You can clone your [**Forked**](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/fork-a-repo) remote git to local and create your PR working branch as a sub-branch of latest [`testing`](https://github.com/abhiTronix/vidgear/tree/testing) branch as follows:

??? info "Functions of different VidGear's Github Branches"

	Following are the base branches for VidGear's code in its Github Repository:

	=== "Master/Main"

		**Branch Features:**

		* Most-Stable 
		* Includes the latest stable-release. 
		* Lacks any latest bug-fixes and changes.
		* Everything CI tested.

		**Cloning:**

		```sh
		# clone your forked repository and `cd` inside
		git clone https://github.com/abhiTronix/vidgear.git && cd vidgear
		```

	=== "Testing"

		**Branch Features:**

		* Stable.
		* Includes latest stable bug-fixes and changes.
		* Used for pushing PR commits. 
		* Everything CI tested.

		**Cloning:**

		```sh
		# clone your forked repository and `cd` inside
		git clone https://github.com/abhiTronix/vidgear.git && cd vidgear
		
		# checkout the latest testing branch
		git checkout testing
		```

	=== "Development"

		**Branch Features:**

		* :warning: Unstable.
		* Includes latest experimental changes.
		* Used for pushing experimental commits. 
		* Nothing CI tested.

		**Cloning:**

		```sh
		# clone your forked repository and `cd` inside
		git clone https://github.com/abhiTronix/vidgear.git && cd vidgear
		
		# checkout the latest development branch
		git checkout development
		```

	**Workflow:** 

	Typically any feature/improvement/bug-fix code flows as follows:

	<figure>
	<img src='../../assets/images/branch_flow.svg'>
	</figure>

```sh
# clone your forked repository(change with your username) and get inside
git clone https://github.com/{YOUR USERNAME}/vidgear.git && cd vidgear

# pull any recent updates
git pull

# checkout the latest testing branch
git checkout testing

# Now create your new branch with suitable name(such as "subbranch_of_testing")
git checkout -b subbranch_of_testing
```

Now after working with this newly created branch for your Pull Request, you can commit and push or merge it locally or remotely as usual.

&nbsp; 

&nbsp; 

## PR Submission Checklist

There are some important checks you need to perform while submitting your Pull Request(s) for VidGear library:

- [x] **Submit a Related Issue:**
  
  * The first thing you do is submit an issue with a [proposal template](https://github.com/abhiTronix/vidgear/issues/new?labels=issue%3A+proposal&template=proposal.md) for your work first and then work on your Pull Request.


- [x] **Submit a Draft Pull Request:**

  * Submit the [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/) from the first day of your development.
  * Add a brief but descriptive title for your PR.
  * Explain what the PR adds, fixes, or improves.
  * In case of bug fixes, add a new unit test case that would fail against your bug fix.
  * Provide output or screenshots, if you can.
  * Make sure your pull request passed all the CI checks _(triggers automatically on pushing commits against testing branch)_. If it's somehow failing, then ask the maintainer for a review.
  * Click "ready for review" when finished.

- [x] **Test, Format & lint code locally:**

  * Make sure to test, format, and lint the modified code locally before every commit. The details are discussed [below ➶](#testing-formatting-linting)

- [x] **Make sensible commit messages:**

  * If your pull request fixes a separate issue number, remember to include `"resolves #issue_number"` in the commit message. Learn more about it [here ➶](https://help.github.com/articles/closing-issues-using-keywords/).
  * Keep the commit message concisely as much as possible at every submit. You can make a supplement to the previous commit with `git commit --amend` command.

- [x] **Perform Integrity Checks:** 

    !!! warning "Any duplicate pull request will be Rejected!"

  * Search GitHub if there's a similar open or closed PR that relates to your submission.
  * Check if your purpose code matches the overall direction, simplicity, and structure of the VidGear and improves it.
  * Retain copyright for your contributions, but also agree to license them for usage by the project and author(s) under the [Apache 2.0 license ➶](https://github.com/abhiTronix/vidgear/blob/master/LICENSE).

- [x] **Link your Issues:**

    !!! tip "For more information on Linking a pull request to an issue, See [this wiki doc➶](https://docs.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue)"

  * Finally, when you're confident enough, make your pull request public. 
  * You can link an issue to a pull request manually or using a supported keyword in the pull request description. It helps collaborators see that someone is working on the issue. 

&nbsp; 

&nbsp; 

## Testing, Formatting & Linting

All Pull Request(s) must be tested, formatted & linted against our library standards as discussed below:

### Requirements

Testing VidGear requires additional test dependencies and dataset, which can be handled manually as follows:

- [x] **Install additional python libraries:**
  
    You can easily install these dependencies via pip:

    ??? info "MPEGDASH for Windows :fontawesome-brands-windows:"
        The [`mpegdash`](https://github.com/sangwonl/python-mpegdash) library has not yet been updated and bugs on windows machines. Therefore install the forked [DEV-version of `mpegdash`](https://github.com/abhiTronix/python-mpegdash) as follows:
        
        ```sh
        python -m pip install https://github.com/abhiTronix/python-mpegdash/releases/download/0.3.0-dev2/mpegdash-0.3.0.dev2-py3-none-any.whl
        ```

    ```sh
    pip install --upgrade six flake8 black pytest pytest-asyncio mpegdash paramiko m3u8 async-asgi-testclient
    ```

- [x] **Download Tests Dataset:** 

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

- [x] **Flake8:** You must run [`flake8`](https://flake8.pycqa.org/en/latest/manpage.html) linting for checking the code base against the coding style (PEP8), programming errors and other cyclomatic complexity:

    ```sh
    flake8 {source_file_or_directory} --count --select=E9,F63,F7,F82 --show-source --statistics
    ```

- [x] **Black:**  Vidgear follows [`black`](https://github.com/psf/black) formatting to make code review faster by producing the smallest diffs possible. You must run it with sensible defaults as follows: 

    ```sh
    black {source_file_or_directory}
    ```

&nbsp; 

&nbsp; 

## Frequently Asked Questions


**Q1. Why do my changes taking so long to be Reviewed and/or Merged?**

!!! info "Submission Aftermaths"
    * After your PR is merged, you can safely delete your branch and pull the changes from the main (upstream) repository.
    * The changes will remain in `testing` branch until next VidGear version is released, then it will be merged into `master` branch.
    * After a successful Merge, your newer contributions will be given priority over others. 

Pull requests will be reviewed by the maintainers and the rationale behind the maintainer’s decision to accept or deny the changes will be posted in the pull request. Please wait for our code review and approval, possibly enhancing your change on request.


**Q2. Would you accept a huge Pull Request with Lots of Changes?**

First, make sure that the changes are somewhat related. Otherwise, please create separate pull requests. Anyway, before submitting a huge change, it's probably a good idea to [open an issue](../../contribution/issue) in the VidGear Github repository to ask the maintainers if they agree with your proposed changes. Otherwise, they could refuse your proposal after you put all that hard work into making the changes. We definitely don't want you to waste your time!

&nbsp; 