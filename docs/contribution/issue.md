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

# Submitting an Issue Guidelines

If you've found a new bug or you've come up with some new feature which can improve the quality of the VidGear, then related issues are welcomed! But, Before you do, please read the following guidelines:

??? question "First Issue on GitHub?" 
    You can easily learn about it from [creating an issue](https://help.github.com/en/github/managing-your-work-on-github/creating-an-issue) wiki.

!!! Info 

    Please note that your issue will be fixed much faster if you spend about half an hour preparing it, including the exact reproduction steps and a demo. If you're in a hurry or don't feel confident, it's fine to report issues with less details, but this makes it less likely they'll get fixed soon.

### Search the Docs and Previous Issues

  * Remember to first search GitHub for a [open or closed issue](https://github.com/abhiTronix/vidgear/issues?q=is%3Aissue) that relates to your submission or already been reported. You may find related information and the discussion might inform you of workarounds that may help to resolve the issue. 
  * For quick questions, please refrain from opening an issue, as you can reach us on [Gitter](https://gitter.im/vidgear/community) community channel.
  * Also, go comprehensively through our dedicated [FAQ & Troubleshooting section](../../help/get_help/#frequently-asked-questions).

### Gather Required Information

* All VidGear APIs provides a `logging` boolean flag in parameters, to log debugged output to terminal. Kindly turn this parameter `True` in the respective API for getting debug output, and paste it with your Issue. 
* In order to reproduce bugs we will systematically ask you to provide a minimal reproduction code for your report. 
* Check and paste, exact VidGear version by running command `#!python python -c "import vidgear; print(vidgear.__version__)"`.

### Follow the Issue Template

* Please format your issue by choosing the appropriate template. 
* Any improper/insufficient reports will be marked with **MISSING : INFORMATION :mag:** and **MISSING : TEMPLATE :grey_question:** like labels, and if we don't hear back from you we may close the issue.

### Raise the Issue

* Add a brief but descriptive title for your issue.
* Keep the issue phrasing in context of the problem.
* Attach source-code/screenshots if you have one.
* Finally, raise it by choosing the appropriate Issue Template: [**Bug report 🐛**](https://github.com/abhiTronix/vidgear/issues/new?assignees=abhiTronix&labels=BUG+%3Abug%3A%2CNEED+TRIAGE+%3Apolice_car%3A&template=bug_report.yaml&title=%5BBug%5D%3A+), [Proposal 💡](https://github.com/abhiTronix/vidgear/issues/new?assignees=&labels=PROPOSAL+%3Aenvelope_with_arrow%3A&template=proposal.yaml&title=%5BProposal%5D%3A+), [Question ❔](https://github.com/abhiTronix/vidgear/issues/new?assignees=&labels=QUESTION+%3Aquestion%3A&template=question.yaml&title=%5BQuestion%5D%3A+).

&nbsp; 