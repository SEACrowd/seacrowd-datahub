# Datasheet Reviewer SOP

Generally, the objective of datasheet review is to ensure that:

### 1. The dataset is available and accessible.
FAQs:
1. Can the dataset be free-upon-request?
    * Yes. For example, we can approve datasets that are hosted on hubs such as HuggingFace, but are gated by required acknowledgements to terms and conditions. The dataset must indicate that it is free-upon-request.

### 2. There must be no duplicate datasheets.
FAQs:
1. What if a contributor submitted a new datasheet for `X` but a datasheet for `X` is already approved?
    * *Is the new datasheet more complete and better than the existing datasheet?*  
      **Yes** → Proceed with the normal review process, change the existing datasheet’s status to “Deprecated”.  
      **No** → Reject.

2. What if more than one contributor submitted new datasheets of the same dataset, and all of them have not yet been approved? (After 23 Nov)
    * Pick one that is relatively better than the others, fix the incorrect/inconsistent parts, then “Approve”.

3. What if more than one contributor submitted new datasheets of the same dataset, and all of them have not yet been approved (Before 23 Nov)
    * Pick one that is more complete than the others, fix any incorrect/inconsistent parts, then “Approve”.
    * For the others, use “sharing points” status.
    * Split the obtained points of the datasheet between the contributors. It doesn’t have to be an equal split. The contributor who gives more complete information can receive higher points.
        * For example, for a datasheet worth 6 points, the assignment could be: contributor 1 gets 3 points, contributor 2 gets 2 points, contributor 3 gets 1 point.
        * We may simplify this by accepting only a subset of identical submissions that have the most complete information and set a ratio split of points (e.g., 7:3).

### 3. The information provided in the datasheet is correct, the information aligns with the dataset, and the dataset is relevant to SEA.
FAQs:
1. What should I do if the datasheet has incorrect or missing information?
    * There are multiple ways to correct this:
        * Ask the contributor to fix it (with some guidance) using the edit link (column AU in the approval sheet).
        * Reviewer uses the edit link (column AU) to fix it themselves.
        * **[NOT RECOMMENDED]** Reviewer uses a hidden sheet (_raw) to directly edit in the cells. This is only recommended when a large number of data per subset of the total dataset must be edited.

2. What should I check and how should I proceed?
    * See the checklist below.

## Approval Checklist
Check the following before approving:
1. Data availability (is it free and open-source or is it private?)
2. Dataset splits (if train, validation, or test are available)
3. Dataset size (in lines, disk size, or any provided metric)
4. Dataset license
5. Task type (whether the data can be represented as the mentioned task)
6. Paper (whether it directs to the correct publication. Archival version has higher priority)
7. Languages (list of all languages it supports)

## What to do next?
1. Change the status to **Rejected**, **Approved**, or **Sharing Points**.
2. Add notes and obtained points (in column BB in the approval sheet)
3. Check the scoring guide and see which languages gets additional points (if any).
4. Add the dataloader name (use Python snake case)
5. Wait for a GitHub issue to be generated for the approved datasheet.
