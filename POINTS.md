# Contribution point guideline

To be considered as a co-author, 20 contribution points are required. To monitor how many points that you have obtained, the contribution point tracking is now live at [this sheet](https://docs.google.com/spreadsheets/d/e/2PACX-1vQDZtJjA6i7JsxS5IlMtVuwOYjr2Pbl_b47yMSH4aAdHDBIpf-CiJQjNQAzcJPEu_aE7kwH4ZvKvPm0/pubhtml?gid=225616890&single=true) and will be updated regularly (although not automatically, yet)!

> **Note**: The purpose of the point system is not to barrier collaboration, but to reward rare and high-quality dataset entries.
We might adjust the point requirement lower to accommodate more co-authorship if needed.

| Contribution type              | Demand              | Points | Max points              | Job description                                                                                                          |
| ------------------------------ | ------------------- | ------ | ----------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Public Datasheet Submission    | As many as possible | 2+bonus      | 6                       | Submit public datasheet via [jotform](https://www.jotform.com/team/232952680898069/seacrowd-sea-datasets)                                                                                   |
| Private Datasheet Submission   | As many as possible | 1      |                         | Submit private datasheet via [jotform](https://www.jotform.com/team/232952680898069/seacrowd-paper-with-private-dataset)                                                                                  |
| Open Access to Private Dataset | As many as possible | 4+bonus     | 10 for the high-quality | Only private dataset owners can do this. Upload the data in a public repository and submit the datasheet in [jotform](https://www.jotform.com/team/232952680898069/seacrowd-sea-datasets). |
| Dataloader Implementation      | As many as possible | 3      | 6 for the hard one      | Implement dataloader based on the respective dataset's schema and task.                                                  |



## Public Datasheet Submission

Submitting a public dataset via [jotform](https://www.jotform.com/team/232952680898069/seacrowd-sea-datasets) is worth a default score of +2. Bonus is applied based on the following conditions:
* modality: +1 pts for speech/vision, +2 pts for others
* language rarity based on https://microsoft.github.io/linguisticdiversity/assets/lang2tax.txt:
   * +0 pts for languages in level 3 or above
   * +1 pts for languages in level 1 and 2
   * +2 pts for languages in level 0 or languages are not mentioned in the list.  

## Private Datasheet Submission

Submitting a private dataset via [jotform](https://www.jotform.com/team/232952680898069/seacrowd-sea-datasets) is worth a default score of +1. 

Providing open access to the private datasets will be granted +4 points, with a bonus applied based on the following conditions:
* modality: +1 pts for speech/vision, +2 pts for others
* language rarity based on https://microsoft.github.io/linguisticdiversity/assets/lang2tax.txt:
   * +0 pts for languages in level 3 or above
   * +1 pts for languages in level 1 and 2
   * +2 pts for languages in level 0 or languages are not mentioned in the list. 

## Implementing Data Loader

Implementing any data loader is granted +3 pts unless otherwise specified on the GitHub issue.
More details [here](DATALOADER.md).
