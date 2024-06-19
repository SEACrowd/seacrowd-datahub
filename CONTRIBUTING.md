# Contributing in SEACrowd

You can contribute by proposing **unregistered NLP dataset** on [our approved record](https://seacrowd.github.io/seacrowd-catalogue/) and our [in-review datasets](https://docs.google.com/spreadsheets/d/1ibbywsC1tQ_sLPX8bUAjC-vrTrUqZgZA46W_sxWw4Ss/edit?usp=sharing). [Just fill out this form](https://jotform.com/team/232952680898069/seacrowd-sea-datasets), and we will check and approve your entry if it meets our requirements (see [this](https://github.com/SEACrowd/seacrowd-datahub/blob/master/REVIEWING.md#approval-checklist) for the detailed checklist).

We will give **contribution points** based on several factors, including: **supported modality**, **language scarcity**, or **task scarcity**.

You can also propose datasets from your past work that have not been released to the public.
In that case, you must first make your dataset open by uploading it publicly, i.e. via Github or Google Drive.

You can submit multiple entries, and if the total **contribution points** is already above the threshold, we will include you as a co-author (Generally it is enough to only propose 1-2 datasets). Read the full method of calculating points [here](POINTS.md).

> **Note**: We are not taking any ownership of the submitted dataset. See FAQ below.

## Any other way to help?

Yes! Aside from new dataset collection, we are also centralizing existing datasets in a single schema that makes it easier for researchers to use Indonesian NLP datasets. You can help us there by building dataset loader. More details about that [here](DATALOADER.md).

Alternatively, we're also listing NLP research papers of Indonesian languages where they do not open their dataset yet. We will contact the authors of these papers later to be involved in SEACrowd. More about this is available in our [Discord server](https://discord.gg/URdhUGsBUQ).

## FAQs

#### Who will be the owner of the submitted dataset?

SEACrowd **do not** make a clone or copy of the submitted dataset. Therefore, the owner of any submitted dataset will remain to the original author. SEACrowd simply build a dataloader, i.e. a file downloader + reader so simplify and standardize the data reading process. We also only collect and centralize metadata of the submitted dataset to be listed in [our catalogue](https://seacrowd.github.io/seacrowd-catalogue/) for better discoverability of your dataset!
Citation to the original data owner is also provided for both SEACrowd and in our catalogue.

#### How can I find the appropriate license for my dataset?

The license for a dataset is not always obvious. Here are some strategies to try in your search,

* check for files such as README or LICENSE that may be distributed with the dataset itself
* check the dataset webpage
* check publications that announce the release of the dataset
* check the website of the organization providing the dataset

If no official license is listed anywhere, but you find a webpage that describes general data usage policies for the dataset, you can fall back to providing that URL in the `_LICENSE` variable. If you can't find any license information, please note in your PR and put `_LICENSE="Unknown"` in your dataset script.

#### What if my dataset is not yet publicly available?
You can upload your dataset publicly first, e.g., on Github. If you're an owner of a Private Dataset that is being contacted by SEACrowd Representative for a possibility of opening that dataset, you may visit this [Private Dataset FAQ](PRIVATE.md).

#### Can I create a PR if I have an idea?

If you have an idea to improve or change the code of the `seacrowd-datahub` repository, please create an `issue` and ask for `feedback` before starting any PRs.

#### I am confused, can you help me?

Yes, you can ask for helps in SEACrowd's community channel! Please join our [Discord server](https://discord.gg/URdhUGsBUQ).

## Thank you!

We greatly appreciate your help!

The artifacts of this initiative will be described in a forthcoming academic paper targeting a machine learning or NLP audience. Please refer to [this section](https://github.com/SEACrowd#how-much-should-i-contribute) for your contribution rewards in helping AI in Southeast Asia. We recognize that some datasets require more effort than others, so please reach out if you have questions. Our goal is to be inclusive with credit!