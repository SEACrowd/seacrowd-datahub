# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import datasets
from copy import deepcopy
from conllu import TokenList

from seacrowd.utils import schemas
from seacrowd.utils.common_parser import load_ud_data, load_ud_data_as_seacrowd_kb
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = r"""
 @misc{11234/1-5287,
 title = {Universal Dependencies 2.13},
 author = {Zeman, Daniel and Nivre, Joakim and Abrams, Mitchell and Ackermann, Elia and Aepli, No{\"e}mi and Aghaei, Hamid and Agi{\'c}, {\v Z}eljko and Ahmadi, Amir and Ahrenberg, Lars and Ajede, Chika Kennedy and Akkurt,
 Salih Furkan and Aleksandravi{\v c}i{\=u}t{\.e}, Gabriel{\.e} and Alfina, Ika and Algom, Avner and Alnajjar, Khalid and Alzetta, Chiara and Andersen, Erik and Antonsen, Lene and Aoyama, Tatsuya and Aplonova, Katya and Aquino, Angelina and Aragon, Carolina and Aranes, Glyd and Aranzabe, Maria Jesus and Ar{\i}can, Bilge Nas and Arnard{\'o}ttir, {\t H}{\'o}runn and Arutie, Gashaw and Arwidarasti, Jessica Naraiswari and Asahara, Masayuki and {\'A}sgeirsd{\'o}ttir, Katla and Aslan, Deniz Baran and Asmazo{\u g}lu, Cengiz and Ateyah, Luma and Atmaca, Furkan and Attia, Mohammed and Atutxa, Aitziber and Augustinus, Liesbeth and Avel{\~a}s, Mariana and Badmaeva, Elena and Balasubramani, Keerthana and Ballesteros, Miguel and Banerjee, Esha and Bank, Sebastian and Barbu Mititelu, Verginica and Barkarson, Starkaður and Basile, Rodolfo and Basmov, Victoria and Batchelor, Colin and Bauer, John and Bedir, Seyyit Talha and Behzad, Shabnam and Belieni, Juan and Bengoetxea, Kepa and Benli, İbrahim and Ben Moshe, Yifat and Berk, G{\"o}zde and Bhat, Riyaz Ahmad and Biagetti, Erica and Bick, Eckhard and Bielinskien{\.e}, Agn{\.e} and Bjarnad{\'o}ttir, Krist{\'{\i}}n and Blokland, Rogier and Bobicev, Victoria and Boizou, Lo{\"{\i}}c and Borges V{\"o}lker, Emanuel and B{\"o}rstell, Carl and Bosco, Cristina and Bouma, Gosse and Bowman, Sam and Boyd, Adriane and Braggaar, Anouck and Branco, Ant{\'o}nio and Brokait{\.e}, Kristina and Burchardt, Aljoscha and Campos, Marisa and Candito, Marie and Caron, Bernard and Caron, Gauthier and Carvalheiro, Catarina and Carvalho, Rita and Cassidy, Lauren and Castro, Maria Clara and Castro, S{\'e}rgio and Cavalcanti, Tatiana and Cebiro{\u g}lu Eryi{\u g}it, G{\"u}l{\c s}en and Cecchini, Flavio Massimiliano and Celano, Giuseppe G. A. and {\v C}{\'e}pl{\"o}, Slavom{\'{\i}}r and Cesur, Neslihan and Cetin, Savas and {\c C}etino{\u g}lu, {\"O}zlem and Chalub, Fabricio and Chamila, Liyanage and Chauhan, Shweta and Chi, Ethan and Chika, Taishi and Cho, Yongseok and Choi, Jinho and Chun, Jayeol and Chung, Juyeon and Cignarella, Alessandra T. and Cinkov{\'a}, Silvie and Collomb, Aur{\'e}lie and {\c C}{\"o}ltekin, {\c C}a{\u g}r{\i} and Connor, Miriam and Corbetta, Claudia and Corbetta, Daniela and Costa, Francisco and Courtin, Marine and Crabb{\'e}, Beno{\^{\i}}t and Cristescu, Mihaela and Cvetkoski, Vladimir and Dale, Ingerid L{\o}yning and Daniel, Philemon and Davidson, Elizabeth and de Alencar, Leonel Figueiredo and Dehouck, Mathieu and de Laurentiis, Martina and de Marneffe, Marie-Catherine and de Paiva, Valeria and Derin, Mehmet Oguz and de Souza, Elvis and Diaz de Ilarraza, Arantza and Dickerson, Carly and Dinakaramani, Arawinda and Di Nuovo, Elisa and Dione, Bamba and Dirix, Peter and Dobrovoljc, Kaja and Doyle, Adrian and Dozat, Timothy and Droganova, Kira and Duran, Magali Sanches and Dwivedi, Puneet and Ebert, Christian and Eckhoff, Hanne and Eguchi, Masaki and Eiche, Sandra and Eli, Marhaba and Elkahky, Ali and Ephrem, Binyam and Erina, Olga and Erjavec, Toma{\v z} and Essaidi, Farah and Etienne, Aline and Evelyn, Wograine and Facundes, Sidney and Farkas, Rich{\'a}rd and Favero, Federica and Ferdaousi, Jannatul and Fernanda, Mar{\'{\i}}lia and Fernandez Alcalde, Hector and Fethi, Amal and Foster, Jennifer and Fransen, Theodorus and Freitas, Cl{\'a}udia and Fujita, Kazunori and Gajdo{\v s}ov{\'a}, Katar{\'{\i}}na and Galbraith, Daniel and Gamba, Federica and Garcia, Marcos and G{\"a}rdenfors, Moa and Gerardi, Fabr{\'{\i}}cio Ferraz and Gerdes, Kim and Gessler, Luke and Ginter, Filip and Godoy, Gustavo and Goenaga, Iakes and Gojenola, Koldo and G{\"o}k{\i}rmak, Memduh and Goldberg, Yoav and G{\'o}mez Guinovart, Xavier and Gonz{\'a}lez Saavedra,
 Berta and Grici{\=u}t{\.e}, Bernadeta and Grioni, Matias and Grobol,
 Lo{\"{\i}}c and Gr{\=
u}z{\={\i}}tis, Normunds and Guillaume, Bruno and Guiller, Kirian and Guillot-Barbance, C{\'e}line and G{\"u}ng{\"o}r, Tunga and Habash, Nizar and Hafsteinsson, Hinrik and Haji{\v c}, Jan and Haji{\v c} jr., Jan and H{\"a}m{\"a}l{\"a}inen, Mika and H{\`a} M{\~y}, Linh and Han, Na-Rae and Hanifmuti, Muhammad Yudistira and Harada, Takahiro and Hardwick, Sam and Harris, Kim and Haug, Dag and Heinecke, Johannes and Hellwig, Oliver and Hennig, Felix and Hladk{\'a}, Barbora and Hlav{\'a}{\v c}ov{\'a}, Jaroslava and Hociung, Florinel and Hohle, Petter and Huang, Yidi and Huerta Mendez, Marivel and Hwang, Jena and Ikeda, Takumi and Ingason, Anton Karl and Ion, Radu and Irimia, Elena and Ishola, {\d O}l{\'a}j{\'{\i}}d{\'e} and Islamaj, Artan and Ito, Kaoru and Jagodzi{\'n}ska, Sandra and Jannat, Siratun and Jel{\'{\i}}nek, Tom{\'a}{\v s} and Jha, Apoorva and Jiang, Katharine and Johannsen, Anders and J{\'o}nsd{\'o}ttir, Hildur and J{\o}rgensen, Fredrik and Juutinen, Markus and Ka{\c s}{\i}kara, H{\"u}ner and Kabaeva, Nadezhda and Kahane, Sylvain and Kanayama, Hiroshi and Kanerva, Jenna and Kara, Neslihan and Karah{\'o}ǧa, Ritv{\'a}n and K{\aa}sen, Andre and Kayadelen, Tolga and Kengatharaiyer, Sarveswaran and Kettnerov{\'a}, V{\'a}clava and Kharatyan, Lilit and Kirchner, Jesse and Klementieva, Elena and Klyachko, Elena and Kocharov, Petr and K{\"o}hn, Arne and K{\"o}ksal, Abdullatif and Kopacewicz, Kamil and Korkiakangas, Timo and K{\"o}se, Mehmet and Koshevoy, Alexey and Kotsyba, Natalia and Kovalevskait{\.e}, Jolanta and Krek, Simon and Krishnamurthy, Parameswari and K{\"u}bler, Sandra and Kuqi, Adrian and Kuyruk{\c c}u, O{\u g}uzhan and Kuzgun, Asl{\i} and Kwak, Sookyoung and Kyle, Kris and Laan, K{\"a}bi and Laippala, Veronika and Lambertino, Lorenzo and Lando, Tatiana and Larasati, Septina Dian and Lavrentiev, Alexei and Lee, John and L{\^e} H{\`{\^o}}ng, Phương and Lenci, Alessandro and Lertpradit, Saran and Leung, Herman and Levina, Maria and Levine, Lauren and Li, Cheuk Ying and Li, Josie and Li, Keying and Li, Yixuan and Li, Yuan and Lim, {KyungTae} and Lima Padovani, Bruna and Lin, Yi-Ju Jessica and Lind{\'e}n, Krister and Liu, Yang Janet and Ljube{\v s}i{\'c}, Nikola and Lobzhanidze, Irina and Loginova, Olga and Lopes, Lucelene and Lusito, Stefano and Luthfi, Andry and Luukko, Mikko and Lyashevskaya, Olga and Lynn, Teresa and Macketanz, Vivien and Mahamdi, Menel and Maillard, Jean and Makarchuk, Ilya and Makazhanov, Aibek and Mandl, Michael and Manning, Christopher and Manurung, Ruli and Mar{\c s}an, B{\"u}{\c s}ra and M{\u a}r{\u a}nduc, C{\u a}t{\u a}lina and Mare{\v c}ek, David and Marheinecke, Katrin and Markantonatou, Stella and Mart{\'{\i}}nez Alonso, H{\'e}ctor and Mart{\'{\i}}n Rodr{\'{\i}}guez, Lorena and Martins, Andr{\'e} and Martins, Cl{\'a}udia and Ma{\v s}ek, Jan and Matsuda, Hiroshi and Matsumoto, Yuji and Mazzei, Alessandro and {McDonald}, Ryan and {McGuinness}, Sarah and Mendon{\c c}a, Gustavo and Merzhevich, Tatiana and Miekka, Niko and Miller, Aaron and Mischenkova, Karina and Missil{\"a}, Anna and Mititelu, C{\u a}t{\u a}lin and Mitrofan, Maria and Miyao, Yusuke and Mojiri Foroushani, {AmirHossein} and Moln{\'a}r, Judit and Moloodi, Amirsaeid and Montemagni, Simonetta and More, Amir and Moreno Romero, Laura and Moretti, Giovanni and Mori, Shinsuke and Morioka, Tomohiko and Moro, Shigeki and Mortensen, Bjartur and Moskalevskyi, Bohdan and Muischnek, Kadri and Munro, Robert and Murawaki, Yugo and M{\"u}{\"u}risep, Kaili and Nainwani, Pinkey and Nakhl{\'e}, Mariam and Navarro Hor{\~n}iacek, Juan Ignacio and Nedoluzhko,
 Anna and Ne{\v s}pore-B{\=e}rzkalne, Gunta and Nevaci, Manuela and Nguy{\~{\^e}}n Th{\d i}, Lương and Nguy{\~{\^e}}n Th{\d i} Minh, Huy{\`{\^e}}n and Nikaido, Yoshihiro and Nikolaev, Vitaly and Nitisaroj, Rattima and Nourian, Alireza and Nunes, Maria das Gra{\c c}as Volpe and Nurmi, Hanna and Ojala, Stina and Ojha, Atul Kr. and {\'O}lad{\'o}ttir, Hulda and Ol{\'u}{\`o}kun, Ad{\'e}day{\d o}̀ and Omura, Mai and Onwuegbuzia, Emeka and Ordan, Noam and Osenova, Petya and {\"O}stling, Robert and {\O}vrelid, Lilja and {\"O}zate{\c s}, {\c S}aziye Bet{\"u}l and {\"O}z{\c c}elik, Merve and {\"O}zg{\"u}r, Arzucan and {\"O}zt{\"u}rk Ba{\c s}aran, Balk{\i}z and Paccosi, Teresa and Palmero Aprosio, Alessio and Panova, Anastasia and Pardo, Thiago Alexandre Salgueiro and Park, Hyunji Hayley and Partanen, Niko and Pascual, Elena and Passarotti, Marco and Patejuk, Agnieszka and Paulino-Passos, Guilherme and Pedonese, Giulia and Peljak-{\L}api{\'n}ska, Angelika and Peng, Siyao and Peng, Siyao Logan and Pereira, Rita and Pereira, S{\'{\i}}lvia and Perez, Cenel-Augusto and Perkova, Natalia and Perrier, Guy and Petrov, Slav and Petrova, Daria and Peverelli, Andrea and Phelan, Jason and Pierre-Louis, Claudel and Piitulainen, Jussi and Pinter, Yuval and Pinto, Clara and Pintucci, Rodrigo and Pirinen, Tommi A and Pitler, Emily and Plamada, Magdalena and Plank, Barbara and Poibeau, Thierry and Ponomareva, Larisa and Popel, Martin and Pretkalni{\c n}a, Lauma and Pr{\'e}vost, Sophie and Prokopidis, Prokopis and Przepi{\'o}rkowski, Adam and Pugh, Robert and Puolakainen, Tiina and Pyysalo, Sampo and Qi, Peng and Querido, Andreia and R{\"a}{\"a}bis, Andriela and Rademaker, Alexandre and Rahoman, Mizanur and Rama, Taraka and Ramasamy, Loganathan and Ramisch, Carlos and Ramos, Joana and Rashel, Fam and Rasooli, Mohammad Sadegh and Ravishankar, Vinit and Real, Livy and Rebeja, Petru and Reddy, Siva and Regnault, Mathilde and Rehm, Georg and Riabi, Arij and Riabov, Ivan and Rie{\ss}ler, Michael and Rimkut{\.e}, Erika and Rinaldi, Larissa and Rituma, Laura and Rizqiyah, Putri and Rocha, Luisa and R{\"o}gnvaldsson, Eir{\'{\i}}kur and Roksandic, Ivan and Romanenko, Mykhailo and Rosa, Rudolf and Roșca, Valentin and Rovati, Davide and Rozonoyer, Ben and Rudina, Olga and Rueter, Jack and R{\'u}narsson, Kristj{\'a}n and Sadde, Shoval and Safari, Pegah and Sahala, Aleksi and Saleh, Shadi and Salomoni, Alessio and Samard{\v z}i{\'c}, Tanja and Samson, Stephanie and Sanguinetti, Manuela and San{\i}yar, Ezgi and S{\"a}rg, Dage and Sartor, Marta and Sasaki,
 Mitsuya and Saul{\={\i}}te, Baiba and Savary, Agata and Sawanakunanon, Yanin and Saxena, Shefali and Scannell, Kevin and Scarlata, Salvatore and Schang, Emmanuel and Schneider, Nathan and Schuster, Sebastian and Schwartz, Lane and Seddah, Djam{\'e} and Seeker, Wolfgang and Seraji, Mojgan and Shahzadi, Syeda and Shen, Mo and Shimada, Atsuko and Shirasu, Hiroyuki and Shishkina, Yana and Shohibussirri, Muh and Shvedova, Maria and Siewert, Janine and Sigurðsson, Einar Freyr and Silva, Jo{\~a}o and Silveira, Aline and Silveira, Natalia and Silveira, Sara and Simi, Maria and Simionescu, Radu and Simk{\'o}, Katalin and {\v S}imkov{\'a}, M{\'a}ria and S{\'{\i}}monarson, Haukur Barri and Simov, Kiril and Sitchinava, Dmitri and Sither, Ted and Skachedubova, Maria and Smith, Aaron and Soares-Bastos, Isabela and Solberg, Per Erik and Sonnenhauser, Barbara and Sourov, Shafi and Sprugnoli, Rachele and Stamou, Vivian and Steingr{\'{\i}}msson, Stein{\t h}{\'o}r and Stella, Antonio and Stephen, Abishek and Straka, Milan and Strickland, Emmett and Strnadov{\'a}, Jana and Suhr, Alane and Sulestio, Yogi Lesmana and Sulubacak, Umut and Suzuki, Shingo and Swanson, Daniel and Sz{\'a}nt{\'o}, Zsolt and Taguchi, Chihiro and Taji, Dima and Tamburini, Fabio and Tan, Mary Ann C. and Tanaka, Takaaki and Tanaya, Dipta and Tavoni, Mirko and Tella, Samson and Tellier, Isabelle and Testori, Marinella and Thomas, Guillaume and Tonelli, Sara and Torga, Liisi and Toska, Marsida and Trosterud, Trond and Trukhina, Anna and Tsarfaty, Reut and T{\"u}rk, Utku and Tyers, Francis and {\t H}{\'o}rðarson, Sveinbj{\"o}rn and {\t H}orsteinsson, Vilhj{\'a}lmur and Uematsu, Sumire and Untilov, Roman and Ure{\v s}ov{\'a}, Zde{\v n}ka and Uria, Larraitz and Uszkoreit, Hans and Utka, Andrius and Vagnoni, Elena and Vajjala, Sowmya and Vak, Socrates and van der Goot, Rob and Vanhove, Martine and van Niekerk, Daniel and van Noord, Gertjan and Varga, Viktor and Vedenina, Uliana and Venturi, Giulia and Villemonte de la Clergerie, Eric and Vincze, Veronika and Vlasova, Natalia and Wakasa, Aya and Wallenberg, Joel C. and Wallin, Lars and Walsh, Abigail and Washington, Jonathan North and Wendt, Maximilan and Widmer, Paul and Wigderson, Shira and Wijono, Sri Hartati and Wille, Vanessa Berwanger and Williams, Seyi and Wir{\'e}n, Mats and Wittern, Christian and Woldemariam, Tsegay and Wong, Tak-sum and Wr{\'o}blewska, Alina and Wu, Qishen and Yako, Mary and Yamashita, Kayo and Yamazaki, Naoki and Yan, Chunxiao and Yasuoka, Koichi and Yavrumyan, Marat M. and Yenice, Arife Bet{\"u}l and Y{\i}ld{\i}z, Olcay Taner and Yu, Zhuoran and Yuliawati, Arlisa and {\v Z}abokrtsk{\'y}, Zden{\v e}k and Zahra, Shorouq and Zeldes, Amir and Zhou, He and Zhu, Hanzhi and Zhu, Yilun and Zhuravleva, Anna and Ziane, Rayan},
 url = {http://hdl.handle.net/11234/1-5287},
 note = {{LINDAT}/{CLARIAH}-{CZ} digital library at the Institute of Formal and Applied Linguistics ({{\'U}FAL}), Faculty of Mathematics and Physics, Charles University},
 copyright = {Licence Universal Dependencies v2.13},
 year = {2023} }
"""

_LANGUAGES = ["ind", "vie", "tgl"]
_LOCAL = False

_DATASETNAME = "ud"

_SUPPORTED_TASKS = [Tasks.POS_TAGGING, Tasks.DEPENDENCY_PARSING, Tasks.MACHINE_TRANSLATION]

#map source subset names to index in `_SUPPORTED_TASKS`
_SOURCE_SUBSETS_TO_TASKS_INDEX = {
    "id_csui": [0,1,2],
    "id_gsd": [0,1],
    "id_pud": [0,2],
    "vi_vtb": [0,1],
    "tl_trg": [0,1,2],
    "tl_ugnayan": [0,2]
}

_DESCRIPTION = """\
Universal Dependencies (UD) is a project that is developing cross-linguistically consistent treebank annotation
 for many languages, with the goal of facilitating multilingual parser development, cross-lingual learning, and 
 parsing research from a language typology perspective. The annotation scheme is based on an evolution of (universal)
   Stanford dependencies (de Marneffe et al., 2006, 2008, 2014), Google universal part-of-speech tags 
   (Petrov et al., 2012), and the Interset interlingua for morphosyntactic tagsets (Zeman, 2008). 
   The general philosophy is to provide a universal inventory of categories and guidelines to facilitate consistent
     annotation of similar constructions across languages, while allowing language-specific extensions when necessary.
"""

_ISO_LANG_MAPPER_UD = {
    "id": "ind",
    "vi": "vie",
    "tl": "tgl"
}

_HOMEPAGE = "https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5287"

_LICENSE = Licenses.APACHE_2_0.value

# "ud-v2.12": "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5150/ud-treebanks-v2.12.tgz?sequence=1&isAllowed=y"
# "ud-v2.13": "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5287/ud-treebanks-v2.13.tgz?sequence=1&isAllowed=y"

_URLS = {
    "ud_id_csui": {
        "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Indonesian-CSUI/master/id_csui-ud-train.conllu",
        "test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Indonesian-CSUI/master/id_csui-ud-test.conllu",
    },
    "ud_id_gsd": {
        "train": "https://raw.githubusercontent.com/indolem/indolem/main/dependency_parsing/UD_Indonesian_GSD/id_gsd-ud-train.conllu",
        "test": "https://raw.githubusercontent.com/indolem/indolem/main/dependency_parsing/UD_Indonesian_GSD/id_gsd-ud-test.conllu",
        "dev": "https://raw.githubusercontent.com/indolem/indolem/main/dependency_parsing/UD_Indonesian_GSD/id_gsd-ud-dev.conllu",
    },
    "ud_id_pud": {
        "test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Indonesian-PUD/master/id_pud-ud-test.conllu"
    },
    "ud_vi_vtb": {
        "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Vietnamese-VTB/master/vi_vtb-ud-train.conllu",
        "test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Vietnamese-VTB/master/vi_vtb-ud-test.conllu",
        "dev": "https://raw.githubusercontent.com/UniversalDependencies/UD_Vietnamese-VTB/master/vi_vtb-ud-dev.conllu",
    },
    "ud_tl_trg": {
        "test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Tagalog-TRG/master/tl_trg-ud-test.conllu",
    },
    "ud_tl_ugnayan": {
        "test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Tagalog-Ugnayan/master/tl_ugnayan-ud-test.conllu",
    },
}

_SOURCE_VERSION = "2.13.0"

_SEACROWD_VERSION = "2024.06.20"


class UDDataset(datasets.GeneratorBasedBuilder):

    SOURCE_BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset_name}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_{subset_name}",
        )
        for subset_name in _SOURCE_SUBSETS_TO_TASKS_INDEX.keys()]
    SEQUENCE_BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset_name}_seacrowd_seq_label",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd Seq Label schema",
            schema="seacrowd_seq_label",
            subset_id=f"{_DATASETNAME}_{subset_name}",
        )
        for subset_name, task_idx in _SOURCE_SUBSETS_TO_TASKS_INDEX.items() if _SUPPORTED_TASKS.index(Tasks.POS_TAGGING) in task_idx]
    KB_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset_name}_seacrowd_kb",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd Knowlegde Base schema",
            schema="seacrowd_kb",
            subset_id=f"{_DATASETNAME}_{subset_name}",
        )
        for subset_name, task_idx in _SOURCE_SUBSETS_TO_TASKS_INDEX.items() if _SUPPORTED_TASKS.index(Tasks.DEPENDENCY_PARSING) in task_idx]
    T2T_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset_name}_seacrowd_t2t",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd Translation T2T schema EN-XX",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}_{subset_name}",
        )
        for subset_name, task_idx in _SOURCE_SUBSETS_TO_TASKS_INDEX.items() if _SUPPORTED_TASKS.index(Tasks.MACHINE_TRANSLATION) in task_idx]

    BUILDER_CONFIGS = SOURCE_BUILDER_CONFIGS + SEQUENCE_BUILDER_CONFIGS + KB_CONFIGS + T2T_CONFIGS

    UPOS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            schema_dict = {
                # metadata
                "sent_id": datasets.Value("string"),
                "text": datasets.Value("string"),
                # tokens
                "id": datasets.Sequence(datasets.Value("string")),
                "form": datasets.Sequence(datasets.Value("string")),
                "lemma": datasets.Sequence(datasets.Value("string")),
                "upos": datasets.Sequence(datasets.Value("string")),
                "xpos": datasets.Sequence(datasets.Value("string")),
                "feats": datasets.Sequence(datasets.Value("string")),
                "head": datasets.Sequence(datasets.Value("string")),
                "deprel": datasets.Sequence(datasets.Value("string")),
                "deps": datasets.Sequence(datasets.Value("string")),
                "misc": datasets.Sequence(datasets.Value("string")),
            }

            # add text_en for UD data that has en text (for T2T)
            if _SUPPORTED_TASKS.index(Tasks.MACHINE_TRANSLATION) in _SOURCE_SUBSETS_TO_TASKS_INDEX["_".join(self.config.subset_id.split("_")[1:])]:
                schema_dict["text_en"] = datasets.Value("string")

            # add "gloss" and "source" for tl_trg subset
            if self.config.subset_id == "ud_tl_trg":
                schema_dict["gloss"] = datasets.Value("string")
                schema_dict["source"] = datasets.Value("string")

            features = datasets.Features(schema_dict)

        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label_features(self.UPOS_TAGS)

        elif self.config.schema == "seacrowd_kb":
            features = schemas.kb_features

        elif self.config.schema == "seacrowd_t2t":
            features = schemas.text2text_features

        else:
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=self._generate_additional_citation(self.config.subset_id),
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        return self._ud_split_generator(dl_manager, self.config.subset_id)
    

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """instance tuple generated in the form (key, labels)"""
        dataset = self._ud_generate_examples(filepath, self.config.subset_id, self.info.features, self.config.schema == "source")

        if self.config.schema == "source":
            pass

        elif self.config.schema == "seacrowd_seq_label":
            #some data has label of "_" which indicates a token that has multiple labels (it has the splitted values in the subsequent/preceeding iterables)
            def remove_invalid_labels_from_seq(sent_id: str, tokens: Iterable, labels: Iterable, invalid_tokens: Iterable):
                _tokens, _labels = [], []
                for idx, val in enumerate(labels):
                    if val not in invalid_tokens:
                        _tokens.append(tokens[idx])
                        _labels.append(labels[idx])
                
                return sent_id, _tokens, _labels

            dataset = list(
                map(
                    lambda d: dict(zip(
                        ("id", "tokens", "labels"),
                        remove_invalid_labels_from_seq(d["sent_id"], d["form"], d["upos"],
                                                     invalid_tokens=("_"))
                    )),
                    filter(lambda d: len(d["form"]) == len(d["upos"]),dataset)
                )
            )
        
        elif self.config.schema == "seacrowd_t2t":
            dataset = list(
                map(
                    lambda d: {
                        "id": d["sent_id"],
                        "text_1": d["text_en"],
                        "text_2": d["text"],
                        "text_1_name": "eng",
                        "text_2_name": _ISO_LANG_MAPPER_UD[self.config.subset_id.split("_")[1]],
                    },
                    filter(lambda d: d.get("text_en"), dataset),
                )
            )

        elif self.config.schema == "seacrowd_kb":
            morph_anomaly = self._get_morph_exceptions(self.config.subset_id)
            dataset = load_ud_data_as_seacrowd_kb(
                        filepath,
                        dataset,
                        morph_exceptions=morph_anomaly
                    )

        else:
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")

        for key, example in enumerate(dataset):
            yield key, example

    @staticmethod
    def _set_load_ud_source_data_kwargs(subset_name: str):

        def _assert_multispan_range_is_one(token_list: TokenList):
            """
            Asserting that all tokens with multiple span can only have 2 span, and \
            no field other than form has important information
            """
            for token in token_list.filter(id=lambda i: not isinstance(i, int)):
                _id = token["id"]
                assert len(_id) == 3, f"Unexpected length of non-int CONLLU Token's id. Expected 3, found {len(_id)};"
                assert all(isinstance(a, b) for a, b in zip(_id, [int, str, int])), f"Non-int ID should be in format of '\\d+-\\d+'. Found {_id};"
                assert _id[2] - _id[0] == 1, f"Token has more than 2 spans. Found {_id[2] - _id[0] + 1} spans;"
                for key in ["lemma", "upos", "xpos", "feats", "head", "deprel", "deps"]:
                    assert token[key] in {"_", None}, f"Field other than 'form' should not contain extra information. Found: '{key}' = '{token[key]}'"

        kwargs_return = {}

        if subset_name == "ud_id_csui":
            kwargs_return = {
                "filter_kwargs": {"id": lambda i: isinstance(i, int)},
                "assert_fn": _assert_multispan_range_is_one}

        if subset_name == "ud_jv_csui":
            kwargs_return = {
                "filter_kwargs": {"id": lambda i: isinstance(i, int)}}

        return kwargs_return

    @staticmethod
    def _generate_additional_citation(subset_name: str):
        # generate additional citation, which `_CITATION` value defined above is appended to the subset-based UD citation

        if subset_name == "ud_id_csui":
            CITATION = r"""
                @article {10.3844/jcssp.2020.1585.1597,
                author = {Alfina, Ika and Budi, Indra and Suhartanto, Heru},
                title = {Tree Rotations for Dependency Trees: Converting the Head-Directionality of Noun Phrases},
                article_type = {journal},
                volume = {16},
                number = {11},
                year = {2020},
                month = {Nov},
                pages = {1585-1597},
                doi = {10.3844/jcssp.2020.1585.1597},
                url = {https://thescipub.com/abstract/jcssp.2020.1585.1597},
                journal = {Journal of Computer Science},
                publisher = {Science Publications}
                }

                """ + _CITATION

        if subset_name == "ud_id_gsd":
            CITATION = r"""
            @inproceedings{mcdonald-etal-2013-universal,
                title = "{U}niversal {D}ependency Annotation for Multilingual Parsing",
                author = {McDonald, Ryan  and
                Nivre, Joakim  and
                Quirmbach-Brundage, Yvonne  and
                Goldberg, Yoav  and
                Das, Dipanjan  and
                Ganchev, Kuzman  and
                Hall, Keith  and
                Petrov, Slav  and
                Zhang, Hao  and
                T{\"a}ckstr{\"o}m, Oscar  and
                Bedini, Claudia  and
                Bertomeu Castell{\'o}, N{\'u}ria  and
                Lee, Jungmee},
                booktitle = "Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
                month = aug,
                year = "2013",
                address = "Sofia, Bulgaria",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/P13-2017",
                pages = "92--97",
            }

            @article{DBLP:journals/corr/abs-2011-00677,
                author    = {Fajri Koto and
                            Afshin Rahimi and
                            Jey Han Lau and
                            Timothy Baldwin},
                title     = {IndoLEM and IndoBERT: {A} Benchmark Dataset and Pre-trained Language
                            Model for Indonesian {NLP}},
                journal   = {CoRR},
                volume    = {abs/2011.00677},
                year      = {2020},
                url       = {https://arxiv.org/abs/2011.00677},
                eprinttype = {arXiv},
                eprint    = {2011.00677},
                timestamp = {Fri, 06 Nov 2020 15:32:47 +0100},
                biburl    = {https://dblp.org/rec/journals/corr/abs-2011-00677.bib},
                bibsource = {dblp computer science bibliography, https://dblp.org}

            """ + _CITATION

        if subset_name == "ud_id_gsd":
            CITATION = r"""
            @conference{2f8c7438a7f44f6b85b773586cff54e8,
                title = "A gold standard dependency treebank for Indonesian",
                author = "Ika Alfina and Arawinda Dinakaramani and Fanany, {Mohamad Ivan} and Heru Suhartanto",
                note = "Publisher Copyright: {\textcopyright} 2019 Proceedings of the 33rd Pacific Asia Conference on Language, Information and Computation, PACLIC 2019. All rights reserved.; \
            33rd Pacific Asia Conference on Language, Information and Computation, PACLIC 2019 ; Conference date: 13-09-2019 Through 15-09-2019",
                year = "2019",
                month = jan,
                day = "1",
                language = "English",
                pages = "1--9",
            }

            @article{DBLP:journals/corr/abs-2011-00677,
                author    = {Fajri Koto and
                            Afshin Rahimi and
                            Jey Han Lau and
                            Timothy Baldwin},
                title     = {IndoLEM and IndoBERT: {A} Benchmark Dataset and Pre-trained Language
                            Model for Indonesian {NLP}},
                journal   = {CoRR},
                volume    = {abs/2011.00677},
                year      = {2020},
                url       = {https://arxiv.org/abs/2011.00677},
                eprinttype = {arXiv},
                eprint    = {2011.00677},
                timestamp = {Fri, 06 Nov 2020 15:32:47 +0100},
                biburl    = {https://dblp.org/rec/journals/corr/abs-2011-00677.bib},
                bibsource = {dblp computer science bibliography, https://dblp.org}
            }

            """ + _CITATION
        
        else:
            CITATION = _CITATION

        return CITATION

    @staticmethod
    def _get_morph_exceptions(subset_name: str):
        morph_anomaly = []
        # not implemented yet
        # if subset_name == "ud_jv_csui":
        #     morph_anomaly = [
        #         # Exceptions due to inconsistencies in the raw data annotation
        #         ("ne", "e"),
        #         ("nipun", "ipun"),
        #         ("me", "e"),  # occurrence word: "Esemme" = "Esem" + "e". original text has double 'm'.
        #     ]

        return morph_anomaly

    @staticmethod
    def _ud_split_generator(dl_manager, subset_name: str):

        split_dset = []
        urls = _URLS[subset_name]
        data_path = dl_manager.download(urls)
        if "train" in data_path:
            split_dset.append(datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_path["train"]
                },
            ))
        if "test" in data_path:
            split_dset.append(datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_path["test"],
                },
            ))
        if "dev" in data_path:
            split_dset.append(datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_path["dev"],
                },
            ))

        return split_dset
    
    @classmethod
    def _ud_generate_examples(cls, filepath: str | list, subset_name: str, features: Iterable, is_source: bool):

        #utility to fill data w/ default val
        def fill_data(data, col_name, fill_val):
            _data = deepcopy(data)
            _data[col_name] = _data.get(col_name, fill_val)
            return _data

        #utility to remove data
        def pop_data(data, col_name):
            _data = deepcopy(data)
            _data.pop(col_name, None)
            return _data

        # allow list of filepath be loaded
        if isinstance(filepath, str):
            filepath = [filepath]

        dataset = []
        for _filepath in filepath:
            dataset.extend(list(
                load_ud_data(
                    _filepath, **cls._set_load_ud_source_data_kwargs(subset_name)
                )
            ))

        # remove the data from source since the occurrence is small (presumably malformed)
        # and not listed in misc features (https://tables.grew.fr/?data=ud_feats/MISC)
        if subset_name == "ud_tl_ugnayan":
            for key in ("newdoc_id", "text_id"):
                dataset = list(map(lambda x: pop_data(x, key), dataset))
        
        if subset_name == "ud_tl_trg":
            for key in ("AP", "BP", "OP", "DP", "PIV"):
                dataset = list(map(lambda x: pop_data(x, key), dataset))

        # fill w/ default only for Source schema
        if is_source:
            for key, default_val in zip(("text_en", "gloss", "source"), ("", "", "")):
                if key in features:
                    dataset = list(map(lambda x: fill_data(x, key, default_val), dataset))
    
        return dataset
