"""
Utility for filtering and loading SEACrowd datasets.
"""
from collections import Counter
from importlib.machinery import SourceFileLoader
import logging
import os
import pathlib
from tqdm import tqdm
from types import ModuleType
from typing import Callable, Iterable, List, Optional, Dict

from dataclasses import dataclass
from dataclasses import field
import datasets

from .utils.configs import SEACrowdConfig
from .utils.constants import Tasks, SCHEMA_TO_TASKS
import pandas as pd

_LARGE_CONFIG_NAMES = [
    'covost2_ind_eng_seacrowd_sptext',
    'covost2_eng_ind_seacrowd_sptext',
    'covost2_ind_eng_seacrowd_t2t',
    'covost2_eng_ind_seacrowd_t2t',
    'cc100_ind_source',
    'cc100_jav_source',
    'cc100_sun_source',
    'cc100_ind_seacrowd_ssp',
    'cc100_jav_seacrowd_ssp',
    'cc100_sun_seacrowd_ssp',
    'indo4b_source', 'indo4b_seacrowd_ssp',
    'indo4b_plus_source', 'indo4b_plus_seacrowd_ssp',
    'kopi_cc_all-raw_source',
    'kopi_cc_all-dedup_source',
    'kopi_cc_all-neardup_source',
    'kopi_cc_all-neardup_clean_source',
    'kopi_cc_2021_10-dedup_source',
    'kopi_cc_2021_10-neardup_source',
    'kopi_cc_2021_10-neardup_clean_source',
    'kopi_cc_2021_17-raw_source',
    'kopi_cc_2021_17-dedup_source',
    'kopi_cc_2021_17-neardup_source',
    'kopi_cc_2021_17-neardup_clean_source',
    'kopi_cc_2021_21-raw_source',
    'kopi_cc_2021_21-dedup_source',
    'kopi_cc_2021_21-neardup_source',
    'kopi_cc_2021_21-neardup_clean_source',
    'kopi_cc_2021_25-raw_source',
    'kopi_cc_2021_25-dedup_source',
    'kopi_cc_2021_25-neardup_source',
    'kopi_cc_2021_25-neardup_clean_source',
    'kopi_cc_2021_31-raw_source',
    'kopi_cc_2021_31-dedup_source',
    'kopi_cc_2021_31-neardup_source',
    'kopi_cc_2021_31-neardup_clean_source',
    'kopi_cc_2021_39-raw_source',
    'kopi_cc_2021_39-dedup_source',
    'kopi_cc_2021_39-neardup_source',
    'kopi_cc_2021_39-neardup_clean_source',
    'kopi_cc_2021_43-raw_source',
    'kopi_cc_2021_43-dedup_source',
    'kopi_cc_2021_43-neardup_source',
    'kopi_cc_2021_43-neardup_clean_source',
    'kopi_cc_2021_49-dedup_source',
    'kopi_cc_2021_49-neardup_source',
    'kopi_cc_2021_49-neardup_clean_source',
    'kopi_cc_2022_05-raw_source',
    'kopi_cc_2022_05-dedup_source',
    'kopi_cc_2022_05-neardup_source',
    'kopi_cc_2022_05-neardup_clean_source',
    'kopi_cc_2022_21-raw_source',
    'kopi_cc_2022_21-dedup_source',
    'kopi_cc_2022_21-neardup_source',
    'kopi_cc_2022_21-neardup_clean_source',
    'kopi_cc_2022_27-raw_source',
    'kopi_cc_2022_27-dedup_source',
    'kopi_cc_2022_27-neardup_source',
    'kopi_cc_2022_27-neardup_clean_source',
    'kopi_cc_all-raw_seacrowd_ssp',
    'kopi_cc_all-dedup_seacrowd_ssp',
    'kopi_cc_all-neardup_seacrowd_ssp',
    'kopi_cc_all-neardup_clean_seacrowd_ssp',
    'kopi_cc_2021_10-dedup_seacrowd_ssp',
    'kopi_cc_2021_10-neardup_seacrowd_ssp',
    'kopi_cc_2021_10-neardup_clean_seacrowd_ssp',
    'kopi_cc_2021_17-raw_seacrowd_ssp',
    'kopi_cc_2021_17-dedup_seacrowd_ssp',
    'kopi_cc_2021_17-neardup_seacrowd_ssp',
    'kopi_cc_2021_17-neardup_clean_seacrowd_ssp',
    'kopi_cc_2021_21-raw_seacrowd_ssp',
    'kopi_cc_2021_21-dedup_seacrowd_ssp',
    'kopi_cc_2021_21-neardup_seacrowd_ssp',
    'kopi_cc_2021_21-neardup_clean_seacrowd_ssp',
    'kopi_cc_2021_25-raw_seacrowd_ssp',
    'kopi_cc_2021_25-dedup_seacrowd_ssp',
    'kopi_cc_2021_25-neardup_seacrowd_ssp',
    'kopi_cc_2021_25-neardup_clean_seacrowd_ssp',
    'kopi_cc_2021_31-raw_seacrowd_ssp',
    'kopi_cc_2021_31-dedup_seacrowd_ssp',
    'kopi_cc_2021_31-neardup_seacrowd_ssp',
    'kopi_cc_2021_31-neardup_clean_seacrowd_ssp',
    'kopi_cc_2021_39-raw_seacrowd_ssp',
    'kopi_cc_2021_39-dedup_seacrowd_ssp',
    'kopi_cc_2021_39-neardup_seacrowd_ssp',
    'kopi_cc_2021_39-neardup_clean_seacrowd_ssp',
    'kopi_cc_2021_43-raw_seacrowd_ssp',
    'kopi_cc_2021_43-dedup_seacrowd_ssp',
    'kopi_cc_2021_43-neardup_seacrowd_ssp',
    'kopi_cc_2021_43-neardup_clean_seacrowd_ssp',
    'kopi_cc_2021_49-dedup_seacrowd_ssp',
    'kopi_cc_2021_49-neardup_seacrowd_ssp',
    'kopi_cc_2021_49-neardup_clean_seacrowd_ssp',
    'kopi_cc_2022_05-raw_seacrowd_ssp',
    'kopi_cc_2022_05-dedup_seacrowd_ssp',
    'kopi_cc_2022_05-neardup_seacrowd_ssp',
    'kopi_cc_2022_05-neardup_clean_seacrowd_ssp',
    'kopi_cc_2022_21-raw_seacrowd_ssp',
    'kopi_cc_2022_21-dedup_seacrowd_ssp',
    'kopi_cc_2022_21-neardup_seacrowd_ssp',
    'kopi_cc_2022_21-neardup_clean_seacrowd_ssp',
    'kopi_cc_2022_27-raw_seacrowd_ssp',
    'kopi_cc_2022_27-dedup_seacrowd_ssp',
    'kopi_cc_2022_27-neardup_seacrowd_ssp',
    'kopi_cc_2022_27-neardup_clean_seacrowd_ssp',
    'kopi_cc_news_2016_source',
    'kopi_cc_news_2017_source',
    'kopi_cc_news_2018_source',
    'kopi_cc_news_2019_source',
    'kopi_cc_news_2020_source',
    'kopi_cc_news_2021_source',
    'kopi_cc_news_2022_source',
    'kopi_cc_news_all_source',
    'kopi_cc_news_2016_seacrowd_ssp',
    'kopi_cc_news_2017_seacrowd_ssp',
    'kopi_cc_news_2018_seacrowd_ssp',
    'kopi_cc_news_2019_seacrowd_ssp',
    'kopi_cc_news_2020_seacrowd_ssp',
    'kopi_cc_news_2021_seacrowd_ssp',
    'kopi_cc_news_2022_seacrowd_ssp',
    'kopi_cc_news_all_seacrowd_ssp',
    'kopi_nllb_all-raw_source',
    'kopi_nllb_all-dedup_source',
    'kopi_nllb_all-neardup_source',
    'kopi_nllb_ace_Latn-raw_source',
    'kopi_nllb_ace_Latn-dedup_source',
    'kopi_nllb_ace_Latn-neardup_source',
    'kopi_nllb_ban_Latn-raw_source',
    'kopi_nllb_ban_Latn-dedup_source',
    'kopi_nllb_ban_Latn-neardup_source',
    'kopi_nllb_bjn_Latn-raw_source',
    'kopi_nllb_bjn_Latn-dedup_source',
    'kopi_nllb_bjn_Latn-neardup_source',
    'kopi_nllb_ind_Latn-raw_source',
    'kopi_nllb_ind_Latn-dedup_source',
    'kopi_nllb_ind_Latn-neardup_source',
    'kopi_nllb_jav_Latn-raw_source',
    'kopi_nllb_jav_Latn-dedup_source',
    'kopi_nllb_jav_Latn-neardup_source',
    'kopi_nllb_min_Latn-raw_source',
    'kopi_nllb_min_Latn-dedup_source',
    'kopi_nllb_min_Latn-neardup_source',
    'kopi_nllb_sun_Latn-raw_source',
    'kopi_nllb_sun_Latn-dedup_source',
    'kopi_nllb_sun_Latn-neardup_source',
    'kopi_nllb_all-raw_seacrowd_ssp',
    'kopi_nllb_all-dedup_seacrowd_ssp',
    'kopi_nllb_all-neardup_seacrowd_ssp',
    'kopi_nllb_ace_Latn-raw_seacrowd_ssp',
    'kopi_nllb_ace_Latn-dedup_seacrowd_ssp',
    'kopi_nllb_ace_Latn-neardup_seacrowd_ssp',
    'kopi_nllb_ban_Latn-raw_seacrowd_ssp',
    'kopi_nllb_ban_Latn-dedup_seacrowd_ssp',
    'kopi_nllb_ban_Latn-neardup_seacrowd_ssp',
    'kopi_nllb_bjn_Latn-raw_seacrowd_ssp',
    'kopi_nllb_bjn_Latn-dedup_seacrowd_ssp',
    'kopi_nllb_bjn_Latn-neardup_seacrowd_ssp',
    'kopi_nllb_ind_Latn-raw_seacrowd_ssp',
    'kopi_nllb_ind_Latn-dedup_seacrowd_ssp',
    'kopi_nllb_ind_Latn-neardup_seacrowd_ssp',
    'kopi_nllb_jav_Latn-raw_seacrowd_ssp',
    'kopi_nllb_jav_Latn-dedup_seacrowd_ssp',
    'kopi_nllb_jav_Latn-neardup_seacrowd_ssp',
    'kopi_nllb_min_Latn-raw_seacrowd_ssp',
    'kopi_nllb_min_Latn-dedup_seacrowd_ssp',
    'kopi_nllb_min_Latn-neardup_seacrowd_ssp',
    'kopi_nllb_sun_Latn-raw_seacrowd_ssp',
    'kopi_nllb_sun_Latn-dedup_seacrowd_ssp',
    'kopi_nllb_sun_Latn-neardup_seacrowd_ssp'
]

_RESOURCE_CONFIG_NAMES = [
    'inset_lexicon_seacrowd_text',
    'kamus_alay_seacrowd_t2t'
]

_CURRENTLY_BROKEN_NAMES = [

]

BENCHMARK_DICT = {
    'IndoNLU': [
        'emot_seacrowd_text',
        'smsa_seacrowd_text',
        'wrete_seacrowd_pairs',
        'casa_seacrowd_text_multi',
        'hoasa_seacrowd_text_multi',
        'facqa_seacrowd_qa',
        'indonlu_nergrit_seacrowd_seq_label',
        'nerp_seacrowd_seq_label',
        'posp_seacrowd_seq_label',
        'term_a_seacrowd_seq_label',
        'keps_seacrowd_seq_label',
        'idn_tagged_corpus_csui_seacrowd_seq_label'
    ],
    'IndoNLG': [
        # MT
        'bible_en_id_seacrowd_t2t',
        'bible_su_id_seacrowd_t2t',
        'bible_jv_id_seacrowd_t2t',
        'ted_en_id_seacrowd_t2t',
        'indo_general_mt_en_id_seacrowd_t2t',
        'news_en_id_seacrowd_t2t',        
        # Summarization
        'indosum_fold0_seacrowd_t2t',
        'liputan6_canonical_seacrowd_t2t',
        'liputan6_xtreme_seacrowd_t2t',
        # Chit Chat
        'xpersona_id_seacrowd_t2t',
        # QA
        'tydiqa_id_seacrowd_qa',
    ],
    'IndoLEM': [
        'indolem_ntp_seacrowd_pairs',
        'indolem_sentiment_seacrowd_text',
        'indolem_ner_ugm_fold0_seacrowd_seq_label',
        'indolem_ner_ugm_fold1_seacrowd_seq_label',
        'indolem_ner_ugm_fold2_seacrowd_seq_label',
        'indolem_ner_ugm_fold3_seacrowd_seq_label',
        'indolem_ner_ugm_fold4_seacrowd_seq_label',
        'indolem_ud_id_gsd_seacrowd_kb',
        'indolem_ud_id_pud_seacrowd_kb',
        'indolem_tweet_ordering_seacrowd_seq_label',
        'indolem_nerui_fold0_seacrowd_seq_label',
        'indolem_nerui_fold1_seacrowd_seq_label',
        'indolem_nerui_fold2_seacrowd_seq_label',
        'indolem_nerui_fold3_seacrowd_seq_label',
        'indolem_nerui_fold4_seacrowd_seq_label'
    ],
    'NusaX': [
        # Ind - XXX
        'nusax_mt_ind_ace_seacrowd_t2t',
        'nusax_mt_ind_ban_seacrowd_t2t',
        'nusax_mt_ind_bjn_seacrowd_t2t',
        'nusax_mt_ind_bug_seacrowd_t2t',
        'nusax_mt_ind_eng_seacrowd_t2t',
        'nusax_mt_ind_jav_seacrowd_t2t',
        'nusax_mt_ind_mad_seacrowd_t2t',
        'nusax_mt_ind_min_seacrowd_t2t',
        'nusax_mt_ind_nij_seacrowd_t2t',
        'nusax_mt_ind_sun_seacrowd_t2t',
        'nusax_mt_ind_bbc_seacrowd_t2t',
        
        # XXX - Ind
        'nusax_mt_ace_ind_seacrowd_t2t',
        'nusax_mt_ban_ind_seacrowd_t2t',
        'nusax_mt_bjn_ind_seacrowd_t2t',
        'nusax_mt_bug_ind_seacrowd_t2t',
        'nusax_mt_eng_ind_seacrowd_t2t',
        'nusax_mt_jav_ind_seacrowd_t2t',
        'nusax_mt_mad_ind_seacrowd_t2t',
        'nusax_mt_min_ind_seacrowd_t2t',
        'nusax_mt_nij_ind_seacrowd_t2t',
        'nusax_mt_sun_ind_seacrowd_t2t',
        'nusax_mt_bbc_ind_seacrowd_t2t',    
    ],    
    'NusaNLU': [
        'emot_seacrowd_text',
        'emotcmt_seacrowd_text',
        'emotion_id_opinion_seacrowd_text',
        'id_abusive_seacrowd_text',
        'id_google_play_review_seacrowd_text',
        'id_google_play_review_posneg_seacrowd_text',
        'id_hatespeech_seacrowd_text',
        'imdb_jv_seacrowd_text',
        'indolem_sentiment_seacrowd_text',
        'jadi_ide_seacrowd_text',
        'nusax_senti_ace_seacrowd_text',
        'nusax_senti_ban_seacrowd_text',
        'nusax_senti_bjn_seacrowd_text',
        'nusax_senti_bug_seacrowd_text',
        'nusax_senti_eng_seacrowd_text',
        'nusax_senti_ind_seacrowd_text',
        'nusax_senti_jav_seacrowd_text',
        'nusax_senti_mad_seacrowd_text',
        'nusax_senti_min_seacrowd_text',
        'nusax_senti_nij_seacrowd_text',
        'nusax_senti_sun_seacrowd_text',
        'nusax_senti_bbc_seacrowd_text',
        'sentiment_nathasa_review_seacrowd_text',
        'smsa_seacrowd_text',    
        'indolem_ntp_seacrowd_pairs',
        'indonli_seacrowd_pairs',
        'code_mixed_jv_id_jv_seacrowd_text',
        'code_mixed_jv_id_id_seacrowd_text',
        'id_am2ico_seacrowd_pairs',
        'id_abusive_news_comment_seacrowd_text',
        'id_hoax_news_seacrowd_text',
        'id_hsd_nofaaulia_seacrowd_text',
        'id_stance_seacrowd_pairs',
        'indo_law_seacrowd_text',
        'indotacos_seacrowd_text',
        'karonese_sentiment_seacrowd_text',
        'su_emot_seacrowd_text',
        'wrete_seacrowd_pairs',
        'id_short_answer_grading_seacrowd_pairs'
    ],    
    'NusaNLG': [
        'bible_en_id_seacrowd_t2t',
        'bible_jv_id_seacrowd_t2t',
        'bible_su_id_seacrowd_t2t',
        'id_panl_bppt_seacrowd_t2t',
        'indo_general_mt_en_id_seacrowd_t2t',
        'indo_religious_mt_en_id_seacrowd_t2t',
        'minangnlp_mt_seacrowd_t2t',
        'news_en_id_seacrowd_t2t',
        'nusax_mt_ace_ind_seacrowd_t2t',
        'nusax_mt_ban_ind_seacrowd_t2t',
        'nusax_mt_bjn_ind_seacrowd_t2t',
        'nusax_mt_bug_ind_seacrowd_t2t',
        'nusax_mt_eng_ind_seacrowd_t2t',
        'nusax_mt_ind_ace_seacrowd_t2t',
        'nusax_mt_ind_ban_seacrowd_t2t',
        'nusax_mt_ind_bjn_seacrowd_t2t',
        'nusax_mt_ind_bug_seacrowd_t2t',
        'nusax_mt_ind_eng_seacrowd_t2t',
        'nusax_mt_ind_jav_seacrowd_t2t',
        'nusax_mt_ind_mad_seacrowd_t2t',
        'nusax_mt_ind_min_seacrowd_t2t',
        'nusax_mt_ind_nij_seacrowd_t2t',
        'nusax_mt_ind_sun_seacrowd_t2t',
        'nusax_mt_ind_bbc_seacrowd_t2t',
        'nusax_mt_jav_ind_seacrowd_t2t',
        'nusax_mt_mad_ind_seacrowd_t2t',
        'nusax_mt_min_ind_seacrowd_t2t',
        'nusax_mt_nij_ind_seacrowd_t2t',
        'nusax_mt_sun_ind_seacrowd_t2t',
        'nusax_mt_bbc_ind_seacrowd_t2t',
        'parallel_su_id_seacrowd_t2t',
        'ted_en_id_seacrowd_t2t',
        'ud_id_csui_seacrowd_t2t',
        'korpus_seacrowd_ind_jav_seacrowd_t2t',
        'korpus_seacrowd_ind_xdy_seacrowd_t2t',
        'korpus_seacrowd_ind_bug_seacrowd_t2t',
        'korpus_seacrowd_ind_sun_seacrowd_t2t',
        'korpus_seacrowd_ind_mad_seacrowd_t2t',
        'korpus_seacrowd_ind_bjn_seacrowd_t2t',
        'korpus_seacrowd_ind_bbc_seacrowd_t2t',
        'korpus_seacrowd_ind_khek_seacrowd_t2t',
        'korpus_seacrowd_ind_msa_seacrowd_t2t',
        'korpus_seacrowd_ind_min_seacrowd_t2t',
        'korpus_seacrowd_ind_tiociu_seacrowd_t2t',
        'korpus_seacrowd_jav_ind_seacrowd_t2t',
        'korpus_seacrowd_xdy_ind_seacrowd_t2t',
        'korpus_seacrowd_bug_ind_seacrowd_t2t',
        'korpus_seacrowd_sun_ind_seacrowd_t2t',
        'korpus_seacrowd_mad_ind_seacrowd_t2t',
        'korpus_seacrowd_bjn_ind_seacrowd_t2t',
        'korpus_seacrowd_bbc_ind_seacrowd_t2t',
        'korpus_seacrowd_khek_ind_seacrowd_t2t',
        'korpus_seacrowd_msa_ind_seacrowd_t2t',
        'korpus_seacrowd_min_ind_seacrowd_t2t',
        'korpus_seacrowd_tiociu_ind_seacrowd_t2t',
        'indosum_fold0_seacrowd_t2t',
        'liputan6_canonical_seacrowd_t2t',
        'xl_sum_seacrowd_t2t',
        'id_qqp_seacrowd_t2t',
        'multilexnorm_seacrowd_t2t',
        'paracotta_id_seacrowd_t2t',
        'stif_indonesia_seacrowd_t2t',        
        'xpersona_id_seacrowd_t2t'
        'facqa_seacrowd_qa',
        'idk_mrc_seacrowd_qa',
        'tydiqa_id_seacrowd_qa'
    ],
    'NusaASR': [
        # Ind
        'indspeech_digit_cdsr_seacrowd_sptext',
        'indspeech_news_lvcsr_seacrowd_sptext',
        'indspeech_teldialog_lvcsr_seacrowd_sptext',
        'indspeech_teldialog_svcsr_seacrowd_sptext',
        'librivox_indonesia_ind_seacrowd_sptext',
        'titml_idn_seacrowd_sptext'
        # Sun
        'indspeech_newstra_ethnicsr_nooverlap_sun_seacrowd_sptext',
        'indspeech_news_ethnicsr_su_nooverlap_seacrowd_sptext',
        'librivox_indonesia_sun_seacrowd_sptext',
        'su_id_asr_seacrowd_sptext',
        # Jav
        'indspeech_newstra_ethnicsr_nooverlap_jav_seacrowd_sptext',
        'indspeech_news_ethnicsr_jv_nooverlap_seacrowd_sptext',
        'librivox_indonesia_jav_seacrowd_sptext',
        'jv_id_asr_seacrowd_sptext',
        # Ban
        'indspeech_newstra_ethnicsr_nooverlap_ban_seacrowd_sptext',
        'librivox_indonesia_ban_seacrowd_sptext',
        # Btk
        'indspeech_newstra_ethnicsr_nooverlap_btk_seacrowd_sptext',
        # Ace
        'librivox_indonesia_ace_seacrowd_sptext',
        # Bug
        'librivox_indonesia_bug_seacrowd_sptext',
        # Min
        'librivox_indonesia_min_seacrowd_sptext',
    ],
    'NusaTranslation': [
        # NusaTranslation Senti
        'nusatranslation_senti_abs_seacrowd_text',
        'nusatranslation_senti_btk_seacrowd_text',
        'nusatranslation_senti_bew_seacrowd_text',
        'nusatranslation_senti_bhp_seacrowd_text',
        'nusatranslation_senti_jav_seacrowd_text',
        'nusatranslation_senti_mad_seacrowd_text',
        'nusatranslation_senti_mak_seacrowd_text',
        'nusatranslation_senti_min_seacrowd_text',
        'nusatranslation_senti_mui_seacrowd_text',
        'nusatranslation_senti_rej_seacrowd_text',
        'nusatranslation_senti_sun_seacrowd_text',

        # NusaTranslation Emot
        'nusatranslation_emot_abs_seacrowd_text',
        'nusatranslation_emot_btk_seacrowd_text',
        'nusatranslation_emot_bew_seacrowd_text',
        'nusatranslation_emot_bhp_seacrowd_text',
        'nusatranslation_emot_jav_seacrowd_text',
        'nusatranslation_emot_mad_seacrowd_text',
        'nusatranslation_emot_mak_seacrowd_text',
        'nusatranslation_emot_min_seacrowd_text',
        'nusatranslation_emot_mui_seacrowd_text',
        'nusatranslation_emot_rej_seacrowd_text',
        'nusatranslation_emot_sun_seacrowd_text',

        # NusaTranslation MT Ind-XXX
        'nusatranslation_mt_ind_abs_seacrowd_t2t',
        'nusatranslation_mt_ind_btk_seacrowd_t2t',
        'nusatranslation_mt_ind_bew_seacrowd_t2t',
        'nusatranslation_mt_ind_bhp_seacrowd_t2t',
        'nusatranslation_mt_ind_jav_seacrowd_t2t',
        'nusatranslation_mt_ind_mad_seacrowd_t2t',
        'nusatranslation_mt_ind_mak_seacrowd_t2t',
        'nusatranslation_mt_ind_min_seacrowd_t2t',
        'nusatranslation_mt_ind_mui_seacrowd_t2t',
        'nusatranslation_mt_ind_rej_seacrowd_t2t',
        'nusatranslation_mt_ind_sun_seacrowd_t2t',

        # NusaTranslation MT XXX-Ind
        'nusatranslation_mt_abs_ind_seacrowd_t2t',
        'nusatranslation_mt_btk_ind_seacrowd_t2t',
        'nusatranslation_mt_bew_ind_seacrowd_t2t',
        'nusatranslation_mt_bhp_ind_seacrowd_t2t',
        'nusatranslation_mt_jav_ind_seacrowd_t2t',
        'nusatranslation_mt_mad_ind_seacrowd_t2t',
        'nusatranslation_mt_mak_ind_seacrowd_t2t',
        'nusatranslation_mt_min_ind_seacrowd_t2t',
        'nusatranslation_mt_mui_ind_seacrowd_t2t',
        'nusatranslation_mt_rej_ind_seacrowd_t2t',
        'nusatranslation_mt_sun_ind_seacrowd_t2t',
    ],
    'NusaParagraph': [
        # NusaParagraph Topic
        'nusaparagraph_topic_btk_seacrowd_text',
        'nusaparagraph_topic_bew_seacrowd_text',
        'nusaparagraph_topic_bug_seacrowd_text',
        'nusaparagraph_topic_jav_seacrowd_text',
        'nusaparagraph_topic_mad_seacrowd_text',
        'nusaparagraph_topic_mak_seacrowd_text',
        'nusaparagraph_topic_min_seacrowd_text',
        'nusaparagraph_topic_mui_seacrowd_text',
        'nusaparagraph_topic_rej_seacrowd_text',
        'nusaparagraph_topic_sun_seacrowd_text',

        # NusaParagraph Rhetoric
        'nusaparagraph_rhetoric_btk_seacrowd_text',
        'nusaparagraph_rhetoric_bew_seacrowd_text',
        'nusaparagraph_rhetoric_bug_seacrowd_text',
        'nusaparagraph_rhetoric_jav_seacrowd_text',
        'nusaparagraph_rhetoric_mad_seacrowd_text',
        'nusaparagraph_rhetoric_mak_seacrowd_text',
        'nusaparagraph_rhetoric_min_seacrowd_text',
        'nusaparagraph_rhetoric_mui_seacrowd_text',
        'nusaparagraph_rhetoric_rej_seacrowd_text',
        'nusaparagraph_rhetoric_sun_seacrowd_text',

        # NusaParagraph Emot
        'nusaparagraph_emot_btk_seacrowd_text',
        'nusaparagraph_emot_bew_seacrowd_text',
        'nusaparagraph_emot_bug_seacrowd_text',
        'nusaparagraph_emot_jav_seacrowd_text',
        'nusaparagraph_emot_mad_seacrowd_text',
        'nusaparagraph_emot_mak_seacrowd_text',
        'nusaparagraph_emot_min_seacrowd_text',
        'nusaparagraph_emot_mui_seacrowd_text',
        'nusaparagraph_emot_rej_seacrowd_text',
        'nusaparagraph_emot_sun_seacrowd_text',
    ],
    'NusaWrites': [
        # NusaTranslation Senti
        'nusatranslation_senti_abs_seacrowd_text',
        'nusatranslation_senti_btk_seacrowd_text',
        'nusatranslation_senti_bew_seacrowd_text',
        'nusatranslation_senti_bhp_seacrowd_text',
        'nusatranslation_senti_jav_seacrowd_text',
        'nusatranslation_senti_mad_seacrowd_text',
        'nusatranslation_senti_mak_seacrowd_text',
        'nusatranslation_senti_min_seacrowd_text',
        'nusatranslation_senti_mui_seacrowd_text',
        'nusatranslation_senti_rej_seacrowd_text',
        'nusatranslation_senti_sun_seacrowd_text',

        # NusaTranslation Emot
        'nusatranslation_emot_abs_seacrowd_text',
        'nusatranslation_emot_btk_seacrowd_text',
        'nusatranslation_emot_bew_seacrowd_text',
        'nusatranslation_emot_bhp_seacrowd_text',
        'nusatranslation_emot_jav_seacrowd_text',
        'nusatranslation_emot_mad_seacrowd_text',
        'nusatranslation_emot_mak_seacrowd_text',
        'nusatranslation_emot_min_seacrowd_text',
        'nusatranslation_emot_mui_seacrowd_text',
        'nusatranslation_emot_rej_seacrowd_text',
        'nusatranslation_emot_sun_seacrowd_text',

        # NusaTranslation MT Ind-XXX
        'nusatranslation_mt_ind_abs_seacrowd_t2t',
        'nusatranslation_mt_ind_btk_seacrowd_t2t',
        'nusatranslation_mt_ind_bew_seacrowd_t2t',
        'nusatranslation_mt_ind_bhp_seacrowd_t2t',
        'nusatranslation_mt_ind_jav_seacrowd_t2t',
        'nusatranslation_mt_ind_mad_seacrowd_t2t',
        'nusatranslation_mt_ind_mak_seacrowd_t2t',
        'nusatranslation_mt_ind_min_seacrowd_t2t',
        'nusatranslation_mt_ind_mui_seacrowd_t2t',
        'nusatranslation_mt_ind_rej_seacrowd_t2t',
        'nusatranslation_mt_ind_sun_seacrowd_t2t',

        # NusaTranslation MT XXX-Ind
        'nusatranslation_mt_abs_ind_seacrowd_t2t',
        'nusatranslation_mt_btk_ind_seacrowd_t2t',
        'nusatranslation_mt_bew_ind_seacrowd_t2t',
        'nusatranslation_mt_bhp_ind_seacrowd_t2t',
        'nusatranslation_mt_jav_ind_seacrowd_t2t',
        'nusatranslation_mt_mad_ind_seacrowd_t2t',
        'nusatranslation_mt_mak_ind_seacrowd_t2t',
        'nusatranslation_mt_min_ind_seacrowd_t2t',
        'nusatranslation_mt_mui_ind_seacrowd_t2t',
        'nusatranslation_mt_rej_ind_seacrowd_t2t',
        'nusatranslation_mt_sun_ind_seacrowd_t2t',

        # NusaParagraph Topic
        'nusaparagraph_topic_btk_seacrowd_text',
        'nusaparagraph_topic_bew_seacrowd_text',
        'nusaparagraph_topic_bug_seacrowd_text',
        'nusaparagraph_topic_jav_seacrowd_text',
        'nusaparagraph_topic_mad_seacrowd_text',
        'nusaparagraph_topic_mak_seacrowd_text',
        'nusaparagraph_topic_min_seacrowd_text',
        'nusaparagraph_topic_mui_seacrowd_text',
        'nusaparagraph_topic_rej_seacrowd_text',
        'nusaparagraph_topic_sun_seacrowd_text',

        # NusaParagraph Rhetoric
        'nusaparagraph_rhetoric_btk_seacrowd_text',
        'nusaparagraph_rhetoric_bew_seacrowd_text',
        'nusaparagraph_rhetoric_bug_seacrowd_text',
        'nusaparagraph_rhetoric_jav_seacrowd_text',
        'nusaparagraph_rhetoric_mad_seacrowd_text',
        'nusaparagraph_rhetoric_mak_seacrowd_text',
        'nusaparagraph_rhetoric_min_seacrowd_text',
        'nusaparagraph_rhetoric_mui_seacrowd_text',
        'nusaparagraph_rhetoric_rej_seacrowd_text',
        'nusaparagraph_rhetoric_sun_seacrowd_text',

        # NusaParagraph Emot
        'nusaparagraph_emot_btk_seacrowd_text',
        'nusaparagraph_emot_bew_seacrowd_text',
        'nusaparagraph_emot_bug_seacrowd_text',
        'nusaparagraph_emot_jav_seacrowd_text',
        'nusaparagraph_emot_mad_seacrowd_text',
        'nusaparagraph_emot_mak_seacrowd_text',
        'nusaparagraph_emot_min_seacrowd_text',
        'nusaparagraph_emot_mui_seacrowd_text',
        'nusaparagraph_emot_rej_seacrowd_text',
        'nusaparagraph_emot_sun_seacrowd_text',
    ],
    'SEACrowd-NLU': [
        # Sentiment Analysis
        "lazada_review_filipino_seacrowd_text",
        "gklmip_sentiment_seacrowd_text",
        "indolem_sentiment_seacrowd_text",
        "id_sentiment_analysis_seacrowd_text",
        "karonese_sentiment_seacrowd_text",
        "wisesight_thai_sentiment_seacrowd_text",
        "wongnai_reviews_seacrowd_text",
        "typhoon_yolanda_tweets_seacrowd_text",
        "smsa_seacrowd_text",
        "prdect_id_sentiment_seacrowd_text",
        "id_sent_emo_mobile_apps_sentiment_seacrowd_text",
        "shopee_reviews_tagalog_seacrowd_text",
        "nusatranslation_senti_abs_seacrowd_text",
        "nusatranslation_senti_btk_seacrowd_text",
        "nusatranslation_senti_bew_seacrowd_text",
        "nusatranslation_senti_bhp_seacrowd_text",
        "nusatranslation_senti_jav_seacrowd_text",
        "nusatranslation_senti_mad_seacrowd_text",
        "nusatranslation_senti_mak_seacrowd_text",
        "nusatranslation_senti_min_seacrowd_text",
        "nusatranslation_senti_mui_seacrowd_text",
        "nusatranslation_senti_rej_seacrowd_text",
        "nusatranslation_senti_sun_seacrowd_text",
        "nusax_senti_ind_seacrowd_text",
        "nusax_senti_ace_seacrowd_text",
        "nusax_senti_jav_seacrowd_text",
        "nusax_senti_sun_seacrowd_text",
        "nusax_senti_min_seacrowd_text",
        "nusax_senti_bug_seacrowd_text",
        "nusax_senti_bbc_seacrowd_text",
        "nusax_senti_ban_seacrowd_text",
        "nusax_senti_nij_seacrowd_text",
        "nusax_senti_mad_seacrowd_text",
        "nusax_senti_bjn_seacrowd_text",
        "nusax_senti_eng_seacrowd_text",
        "indonglish_seacrowd_text",
        # Topic Analysis
        "gklmip_newsclass_seacrowd_text",
        "indonesian_news_dataset_seacrowd_text",
        "uit_vion_seacrowd_text",
        "sib_200_ace_Arab_seacrowd_text",
        "sib_200_ace_Latn_seacrowd_text",
        "sib_200_ban_Latn_seacrowd_text",
        "sib_200_bjn_Arab_seacrowd_text",
        "sib_200_bjn_Latn_seacrowd_text",
        "sib_200_bug_Latn_seacrowd_text",
        "sib_200_ceb_Latn_seacrowd_text",
        "sib_200_ilo_Latn_seacrowd_text",
        "sib_200_ind_Latn_seacrowd_text",
        "sib_200_jav_Latn_seacrowd_text",
        "sib_200_kac_Latn_seacrowd_text",
        "sib_200_khm_Khmr_seacrowd_text",
        "sib_200_lao_Laoo_seacrowd_text",
        "sib_200_lus_Latn_seacrowd_text",
        "sib_200_min_Arab_seacrowd_text",
        "sib_200_min_Latn_seacrowd_text",
        "sib_200_mya_Mymr_seacrowd_text",
        "sib_200_pag_Latn_seacrowd_text",
        "sib_200_shn_Mymr_seacrowd_text",
        "sib_200_sun_Latn_seacrowd_text",
        "sib_200_tgl_Latn_seacrowd_text",
        "sib_200_tha_Thai_seacrowd_text",
        "sib_200_vie_Latn_seacrowd_text",
        "sib_200_war_Latn_seacrowd_text",
        "sib_200_zsm_Latn_seacrowd_text",
        "nusaparagraph_topic_btk_seacrowd_text",
        "nusaparagraph_topic_bew_seacrowd_text",
        "nusaparagraph_topic_bug_seacrowd_text",
        "nusaparagraph_topic_jav_seacrowd_text",
        "nusaparagraph_topic_mad_seacrowd_text",
        "nusaparagraph_topic_mak_seacrowd_text",
        "nusaparagraph_topic_min_seacrowd_text",
        "nusaparagraph_topic_mui_seacrowd_text",
        "nusaparagraph_topic_rej_seacrowd_text",
        "nusaparagraph_topic_sun_seacrowd_text",
        # Reasoning
        "emotes_3k_tgl_seacrowd_text",
        "emotes_3k_eng_seacrowd_text",
        "indo_story_cloze_seacrowd_qa",
        "xstorycloze_id_seacrowd_qa",
        "xstorycloze_my_seacrowd_qa",
        # Standard Testing QA
        "indommlu_ind_seacrowd_qa",
        "indommlu_ban_seacrowd_qa",
        "indommlu_mad_seacrowd_qa",
        "indommlu_mak_seacrowd_qa",
        "indommlu_sun_seacrowd_qa",
        "indommlu_jav_seacrowd_qa",
        "indommlu_bjn_seacrowd_qa",
        "indommlu_abl_seacrowd_qa",
        "indommlu_nij_seacrowd_qa",
        "seaeval_cross_mmlu_ind_seacrowd_qa",
        "seaeval_cross_mmlu_vie_seacrowd_qa",
        "seaeval_cross_mmlu_zlm_seacrowd_qa",
        "seaeval_cross_mmlu_fil_seacrowd_qa",
        "seaeval_cross_logiqa_ind_seacrowd_qa",
        "seaeval_cross_logiqa_vie_seacrowd_qa",
        "seaeval_cross_logiqa_zlm_seacrowd_qa",
        "seaeval_cross_logiqa_fil_seacrowd_qa",
        "m3exam_jav_seacrowd_qa",
        "m3exam_tha_seacrowd_qa",
        "m3exam_vie_seacrowd_qa",
        "okapi_m_arc_ind_seacrowd_qa",
        "okapi_m_arc_vie_seacrowd_qa",
        # Cultural QA
        "copal_colloquial_seacrowd_qa",
        "xcopa_tha_seacrowd_qa",
        "xcopa_vie_seacrowd_qa",
        "xcopa_ind_seacrowd_qa",
        "seaeval_sg_eval_eng_seacrowd_qa",
        "seaeval_ph_eval_eng_seacrowd_qa",
        "mabl_ind_seacrowd_qa",
        "mabl_jav_seacrowd_qa",
        "mabl_sun_seacrowd_qa",
        # Reading Comprehension QA
        "belebele_ceb_latn_seacrowd_qa",
        "belebele_ilo_latn_seacrowd_qa",
        "belebele_ind_latn_seacrowd_qa",
        "belebele_jav_latn_seacrowd_qa",
        "belebele_kac_latn_seacrowd_qa",
        "belebele_khm_khmr_seacrowd_qa",
        "belebele_lao_laoo_seacrowd_qa",
        "belebele_mya_mymr_seacrowd_qa",
        "belebele_shn_mymr_seacrowd_qa",
        "belebele_sun_latn_seacrowd_qa",
        "belebele_tgl_latn_seacrowd_qa",
        "belebele_tha_thai_seacrowd_qa",
        "belebele_vie_latn_seacrowd_qa",
        "belebele_war_latn_seacrowd_qa",
        "belebele_zsm_latn_seacrowd_qa",
        # NLI
        "indonli_seacrowd_pairs",
        "wrete_seacrowd_pairs",
        "snli_indo_seacrowd_pairs",
        "myxnli_seacrowd_pairs",
        "xnli.tha_seacrowd_pairs",
        "xnli.vie_seacrowd_pairs",
    ],
    "SEACrowd-NLG": [
        # SUMMARIZATION
        "lr_sum_ind_seacrowd_t2t",
        "lr_sum_vie_seacrowd_t2t",
        "lr_sum_lao_seacrowd_t2t",
        "lr_sum_tha_seacrowd_t2t",
        "lr_sum_khm_seacrowd_t2t",
        "lr_sum_mya_seacrowd_t2t",
        "xl_sum_mya_seacrowd_t2t",
        "xl_sum_ind_seacrowd_t2t",
        "xl_sum_tha_seacrowd_t2t",
        "xl_sum_vie_seacrowd_t2t",
        # MACHINE TRANSLATION
        "lio_and_central_flores_eng_ljl_seacrowd_t2t",
        "lio_and_central_flores_ljl_eng_seacrowd_t2t",
        "flores200_eng_Latn_ace_Latn_seacrowd_t2t",
        "flores200_eng_Latn_ban_Latn_seacrowd_t2t",
        "flores200_eng_Latn_bjn_Latn_seacrowd_t2t",
        "flores200_eng_Latn_bug_Latn_seacrowd_t2t",
        "flores200_eng_Latn_ceb_Latn_seacrowd_t2t",
        "flores200_eng_Latn_ilo_Latn_seacrowd_t2t",
        "flores200_eng_Latn_ind_Latn_seacrowd_t2t",
        "flores200_eng_Latn_jav_Latn_seacrowd_t2t",
        "flores200_eng_Latn_kac_Latn_seacrowd_t2t",
        "flores200_eng_Latn_khm_Khmr_seacrowd_t2t",
        "flores200_eng_Latn_lao_Laoo_seacrowd_t2t",
        "flores200_eng_Latn_lus_Latn_seacrowd_t2t",
        "flores200_eng_Latn_min_Latn_seacrowd_t2t",
        "flores200_eng_Latn_mya_Mymr_seacrowd_t2t",
        "flores200_eng_Latn_pag_Latn_seacrowd_t2t",
        "flores200_eng_Latn_shn_Mymr_seacrowd_t2t",
        "flores200_eng_Latn_sun_Latn_seacrowd_t2t",
        "flores200_eng_Latn_tha_Thai_seacrowd_t2t",
        "flores200_eng_Latn_vie_Latn_seacrowd_t2t",
        "flores200_eng_Latn_war_Latn_seacrowd_t2t",
        "flores200_eng_Latn_zsm_Latn_seacrowd_t2t",
        "ntrex_128_eng-US_ind_seacrowd_t2t",
        "ntrex_128_eng-US_mya_seacrowd_t2t",
        "ntrex_128_eng-US_fil_seacrowd_t2t",
        "ntrex_128_eng-US_khm_seacrowd_t2t",
        "ntrex_128_eng-US_lao_seacrowd_t2t",
        "ntrex_128_eng-US_zlm_seacrowd_t2t",
        "ntrex_128_eng-US_tha_seacrowd_t2t",
        "ntrex_128_eng-US_vie_seacrowd_t2t",
        "ntrex_128_eng-US_hmv_seacrowd_t2t",
        "nusax_mt_eng_ind_seacrowd_t2t",
        "nusax_mt_eng_ace_seacrowd_t2t",
        "nusax_mt_eng_jav_seacrowd_t2t",
        "nusax_mt_eng_sun_seacrowd_t2t",
        "nusax_mt_eng_min_seacrowd_t2t",
        "nusax_mt_eng_bug_seacrowd_t2t",
        "nusax_mt_eng_bbc_seacrowd_t2t",
        "nusax_mt_eng_ban_seacrowd_t2t",
        "nusax_mt_eng_nij_seacrowd_t2t",
        "nusax_mt_eng_mad_seacrowd_t2t",
        "nusax_mt_eng_bjn_seacrowd_t2t",
        "flores200_ace_Latn_eng_Latn_seacrowd_t2t",
        "flores200_ban_Latn_eng_Latn_seacrowd_t2t",
        "flores200_bjn_Latn_eng_Latn_seacrowd_t2t",
        "flores200_bug_Latn_eng_Latn_seacrowd_t2t",
        "flores200_ceb_Latn_eng_Latn_seacrowd_t2t",
        "flores200_ilo_Latn_eng_Latn_seacrowd_t2t",
        "flores200_ind_Latn_eng_Latn_seacrowd_t2t",
        "flores200_jav_Latn_eng_Latn_seacrowd_t2t",
        "flores200_kac_Latn_eng_Latn_seacrowd_t2t",
        "flores200_khm_Khmr_eng_Latn_seacrowd_t2t",
        "flores200_lao_Laoo_eng_Latn_seacrowd_t2t",
        "flores200_lus_Latn_eng_Latn_seacrowd_t2t",
        "flores200_min_Latn_eng_Latn_seacrowd_t2t",
        "flores200_mya_Mymr_eng_Latn_seacrowd_t2t",
        "flores200_pag_Latn_eng_Latn_seacrowd_t2t",
        "flores200_shn_Mymr_eng_Latn_seacrowd_t2t",
        "flores200_sun_Latn_eng_Latn_seacrowd_t2t",
        "flores200_tha_Thai_eng_Latn_seacrowd_t2t",
        "flores200_vie_Latn_eng_Latn_seacrowd_t2t",
        "flores200_war_Latn_eng_Latn_seacrowd_t2t",
        "flores200_zsm_Latn_eng_Latn_seacrowd_t2t",
        "ntrex_128_ind_eng-US_seacrowd_t2t",
        "ntrex_128_mya_eng-US_seacrowd_t2t",
        "ntrex_128_fil_eng-US_seacrowd_t2t",
        "ntrex_128_khm_eng-US_seacrowd_t2t",
        "ntrex_128_lao_eng-US_seacrowd_t2t",
        "ntrex_128_zlm_eng-US_seacrowd_t2t",
        "ntrex_128_tha_eng-US_seacrowd_t2t",
        "ntrex_128_vie_eng-US_seacrowd_t2t",
        "ntrex_128_hmv_eng-US_seacrowd_t2t",
        "nusax_mt_ace_eng_seacrowd_t2t",
        "nusax_mt_jav_eng_seacrowd_t2t",
        "nusax_mt_sun_eng_seacrowd_t2t",
        "nusax_mt_min_eng_seacrowd_t2t",
        "nusax_mt_bug_eng_seacrowd_t2t",
        "nusax_mt_bbc_eng_seacrowd_t2t",
        "nusax_mt_ban_eng_seacrowd_t2t",
        "nusax_mt_nij_eng_seacrowd_t2t",
        "nusax_mt_mad_eng_seacrowd_t2t",
        "nusax_mt_bjn_eng_seacrowd_t2t",
        # EXTRACTIVE ABSTRACTIVE QA
        "facqa_seacrowd_qa",
        "iapp_squad_seacrowd_qa",
        "qasina_seacrowd_qa",
        "mkqa_khm_seacrowd_qa", 
        "mkqa_zsm_seacrowd_qa",
        "mkqa_tha_seacrowd_qa",
        "mkqa_vie_seacrowd_qa"
    ],
    "SEACrowd-Speech": [
        'asr_ibsc_seacrowd_sptext',
        'commonvoice_120_tha_seacrowd_sptext',
        'commonvoice_120_vie_seacrowd_sptext',
        'commonvoice_120_ind_seacrowd_sptext',
        'commonvoice_120_cnh_seacrowd_sptext',
        'fleurs_mya_seacrowd_sptext',
        'fleurs_ceb_seacrowd_sptext',
        'fleurs_fil_seacrowd_sptext',
        'fleurs_ind_seacrowd_sptext',
        'fleurs_jav_seacrowd_sptext',
        'fleurs_khm_seacrowd_sptext',
        'fleurs_lao_seacrowd_sptext',
        'fleurs_zlm_seacrowd_sptext',
        'fleurs_tha_seacrowd_sptext',
        'fleurs_vie_seacrowd_sptext',
        'indspeech_newstra_ethnicsr_nooverlap_sun_seacrowd_sptext',
        'indspeech_newstra_ethnicsr_nooverlap_jav_seacrowd_sptext',
        'indspeech_newstra_ethnicsr_nooverlap_ban_seacrowd_sptext',
        'indspeech_newstra_ethnicsr_nooverlap_btk_seacrowd_sptext',
    ],
    "SEACrowd-VL": [
        "xm3600_fil_seacrowd_imtext",
        "xm3600_id_seacrowd_imtext",
        "xm3600_th_seacrowd_imtext",
        "xm3600_vi_seacrowd_imtext",
    ],
}

@dataclass
class SEACrowdMetadata:
    """Metadata for one config of a dataset."""

    script: pathlib.Path
    dataset_name: str
    tasks: List[Tasks]
    languages: List[str]
    config: SEACrowdConfig
    is_local: bool
    is_seacrowd_schema: bool
    seacrowd_schema_caps: Optional[str]
    is_large: bool
    is_resource: bool
    is_default: bool
    is_broken: bool
    seacrowd_version: str
    source_version: str
    citation: str
    description: str
    homepage: str
    license: str

    _ds_module: datasets.load.DatasetModule = field(repr=False)
    _py_module: ModuleType = field(repr=False)
    _ds_cls: type = field(repr=False)

    def get_load_dataset_kwargs(
        self,
        **extra_load_dataset_kwargs,
    ):
        return {
            "path": self.script,
            "name": self.config.name,
            **extra_load_dataset_kwargs,
        }

    def load_dataset(
        self,
        trust_remote_code=True,
        **extra_load_dataset_kwargs,
    ):
        return datasets.load_dataset(
            path=self.script,
            name=self.config.name,
            trust_remote_code=trust_remote_code,
            **extra_load_dataset_kwargs,
        )

    def get_metadata(self, **extra_load_dataset_kwargs):
        if not self.is_seacrowd_schema:
            raise ValueError("only supported for seacrowd schemas")
        dsd = self.load_dataset(**extra_load_dataset_kwargs)
        split_metas = {}
        for split, ds in dsd.items():
            meta = SCHEMA_TO_METADATA_CLS[self.config.schema].from_dataset(ds)
            split_metas[split] = meta
        return split_metas


def default_is_keeper(metadata: SEACrowdMetadata) -> bool:
    return not metadata.is_large and not metadata.is_resource and metadata.is_seacrowd_schema

class SEACrowdConfigHelper:
    """
    Handles creating and filtering SEACrowdMetadata instances.
    """

    def __init__(
        self,
        helpers: Optional[Iterable[SEACrowdMetadata]] = None,
        keep_broken: bool = False,
        trust_remote_code: bool = True,
    ):

        path_to_here = pathlib.Path(__file__).parent.absolute()
        self.path_to_sea_datasets = (path_to_here / "sea_datasets").resolve()
        self.dataloader_scripts = sorted(
            self.path_to_sea_datasets.glob(os.path.join("*", "*.py"))
        )
        self.dataloader_scripts = [
            el for el in self.dataloader_scripts if el.name != "__init__.py"
        ]

        # if helpers are passed in, just attach and go
        if helpers is not None:
            if keep_broken:
                self._helpers = helpers
            else:
                self._helpers = [helper for helper in helpers if not helper.is_broken]
            return

        # otherwise, create all helpers available in package
        helpers = []
        for dataloader_script in tqdm(self.dataloader_scripts):
            if dataloader_script.stem != dataloader_script.parent.stem:
                continue
            dataset_name = dataloader_script.stem
            py_module = SourceFileLoader(
                dataset_name, dataloader_script.as_posix()
            ).load_module()
            ds_module = datasets.load.dataset_module_factory(
                dataloader_script.as_posix(), trust_remote_code=trust_remote_code,
            )
            ds_cls = datasets.load.import_main_class(ds_module.module_path)

            for config in ds_cls.BUILDER_CONFIGS:

                is_seacrowd_schema = config.schema.startswith("seacrowd")
                if is_seacrowd_schema:
                    seacrowd_schema_caps = '_'.join(config.schema.split("_")[1:]).upper()
                    tasks = SCHEMA_TO_TASKS[seacrowd_schema_caps] & set(
                        py_module._SUPPORTED_TASKS
                    )
                else:
                    tasks = py_module._SUPPORTED_TASKS
                    seacrowd_schema_caps = None

                helpers.append(
                    SEACrowdMetadata(
                        script=dataloader_script.as_posix(),
                        dataset_name=dataset_name,
                        tasks=tasks,
                        languages=py_module._LANGUAGES,
                        config=config,
                        is_local=py_module._LOCAL,
                        is_seacrowd_schema=is_seacrowd_schema,
                        seacrowd_schema_caps=seacrowd_schema_caps,
                        is_large=config.name in _LARGE_CONFIG_NAMES,
                        is_resource=config.name in _RESOURCE_CONFIG_NAMES,
                        is_default=config.name == ds_cls.DEFAULT_CONFIG_NAME,
                        is_broken=config.name in _CURRENTLY_BROKEN_NAMES,
                        seacrowd_version=py_module._SEACROWD_VERSION,
                        source_version=py_module._SOURCE_VERSION,
                        citation=py_module._CITATION,
                        description=py_module._DESCRIPTION,
                        homepage=py_module._HOMEPAGE,
                        license=py_module._LICENSE,
                        _ds_module=ds_module,
                        _py_module=py_module,
                        _ds_cls=ds_cls,
                    )
                )

        if keep_broken:
            self._helpers = helpers
        else:
            self._helpers = [helper for helper in helpers if not helper.is_broken]

    def available_dataset_names(self, schema=None) -> List[str]:
        if schema == None:
            return sorted(list(set([helper.dataset_name for helper in self])))
        elif "seacrowd" in schema or "source" in schema:
            return sorted(list(set([helper.dataset_name for helper in self if schema in helper.config.schema])))
        else:
            raise ValueError("Schema must be either variations of `seacrowd` or `source`.")
    
    def available_config_names(self, dataset_name=None) -> List[str]:
        if dataset_name == None:
            return sorted(list(set([helper.config.name for helper in self])))
        else:
            return sorted(list(set([helper.config.name for helper in self if helper.dataset_name == dataset_name])))

    def for_dataset(self, dataset_name: str, schema: str) -> "SEACrowdMetadata":
        # Widening the search for suggestions (if needed)
        other_helpers = [helper for helper in self if helper.dataset_name == dataset_name and schema in helper.config.name]
        # The returned helpers should match the `schema`
        helpers = [helper for helper in other_helpers if schema in helper.config.name]

        if len(helpers) == 0:
            error_msg = f"No helper with helper.dataset_name = {dataset_name}"
            if len(other_helpers) > 0:
                error_msg += f" with schemas = {schema}. Available schemas are: {[helper.config.schema for helper in other_helpers]}"
            raise ValueError(f"{error_msg}.")
        
        elif len(helpers) > 1:
            error_msg = f"Multiple helpers with helper.dataset_name = {dataset_name}"
            schema_list = [helper.config.schema for helper in helpers]
            unique_schemas = list(set(schema_list))
            # If there are duplicates in the schema ids
            # Commonly occurred in MT datasets (nusax_mt_ind_ace_seacrowd_t2t, nusax_mt_ind_ban_seacrowd_t2t, etc.)
            if len(unique_schemas) < len(schema_list):
                error_msg += f". Better use `load_dataset_by_config_name` or `load_datasets_by_config_names` because there are multiple configs with identical schemas: {[helper.config.name for helper in helpers]}"
            else:
                error_msg += f". Specify `schema` as either: {unique_schemas}"
            raise ValueError(f"{error_msg}.")
        
        return helpers[0]
    
    def for_datasets(self, dataset_names: list[str], schema: str) -> "SEACrowdMetadata":
        helpers = [self.for_dataset(dataset_name, schema=schema) for dataset_name in dataset_names]
        return helpers

    def for_config_name(self, config_name: str) -> "SEACrowdMetadata":
        helpers = [helper for helper in self if helper.config.name == config_name]

        if len(helpers) == 0:
            raise ValueError(f"No helper with helper.config.name = {config_name}.")
        elif len(helpers) > 1:
            raise ValueError(
                f"Multiple helpers with helper.config.name = {config_name}."
            )
        return helpers[0]
    
    def for_config_names(self, config_names: list[str]) -> "SEACrowdMetadata":
        helpers = [helper for helper in self if helper.config.name in config_names]
        
        if len(helpers) == 0:
            raise ValueError(f"No helper with helper.config.name = {config_names}.")
        return helpers

    def default_for_dataset(self, dataset_name: str) -> "SEACrowdMetadata":
        helpers = [
            helper
            for helper in self
            if helper.is_default and helper.dataset_name == dataset_name
        ]
        assert len(helpers) == 1
        return helpers[0]

    # def filtered(
    #     self, is_keeper: Callable[[SEACrowdMetadata], bool]
    # ) -> "SEACrowdConfigHelper":
    #     """Return dataset config helpers that match is_keeper."""
    #     return SEACrowdConfigHelper(
    #         helpers=[helper for helper in self if is_keeper(helper)]
    #     )

    def __repr__(self):
        # return "\n\n".join([helper.__repr__() for helper in self])
        return f"SEACrowd Config Helper. Datasets include: {self.available_dataset_names}."

    def __str__(self):
        return self.__repr__()
        # # return f"SEACrowd Config Helper. Datasets include: {self.available_dataset_names}."
        # return "SEACrowd Config Helper."

    def __iter__(self):
        for helper in self._helpers:
            yield helper

    def __len__(self):
        return len(self._helpers)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return SEACrowdConfigHelper(
                helpers=[self._helpers[ii] for ii in range(start, stop, step)]
            )
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"The index ({key}) is out of range.")
            return self._helpers[key]
        else:
            raise TypeError("Invalid argument type.")
    
    def list_datasets(self, with_config=False):
        name_to_schema = {}
        for helper in self:
            if helper.dataset_name not in name_to_schema:
                name_to_schema[helper.dataset_name] = []
            name_to_schema[helper.dataset_name].append(helper.config.name)
        if not with_config:
            return list(name_to_schema.keys())
        else:
            return name_to_schema
    
    def load_dataset(self, dataset_name, schema='seacrowd'):
        helper = self.for_dataset(dataset_name, schema=schema)
        return helper.load_dataset()

    def load_datasets(self, dataset_names, schema='seacrowd'):
        helpers = self.for_datasets(dataset_names, schema=schema)
        return [helper.load_dataset() for helper in helpers]
    
    def load_dataset_by_config_name(self, config_name):
        helper = self.for_config_name(config_name)
        return helper.load_dataset()

    def load_datasets_by_config_names(self, config_names):
        helpers = self.for_config_name(config_names)
        return [helper.load_dataset() for helper in helpers]
        
    def list_benchmarks(self):
        return list(BENCHMARK_DICT.keys())

    def load_benchmark(self, benchmark_name):
        config_list = BENCHMARK_DICT[benchmark_name]
        helpers = self.for_config_names(config_list)
        return [helper.load_dataset() for helper in helpers]

# Metadata Helper
@dataclass
class MetaDict:
    data: dict = None
    
class SEACrowdMetadataHelper:
    """
    Handles creating and filtering SEACrowdMetadata instances.
    """

    def __init__(
        self,
        meta_df: Optional[pd.DataFrame] = None,
        keep_broken: bool = False
    ):
        # Load Config Helper
        self._conhelps = SEACrowdConfigHelper()
        
        # if meta_df are passed in, just attach and go
        if meta_df is not None:
            if keep_broken:
                self._meta_df = meta_df
            else:
                self._meta_df = meta_df[~meta_df.is_broken]
            return
        
        # Load Metadata
        self._meta_df = pd.read_csv('https://docs.google.com/spreadsheets/d/17o83IvWxmtGLYridZis0nEprHhsZIMeFtHGtXV35h6M/export?format=csv&gid=879729812', skiprows=1)
        self._meta_df = self._meta_df[self._meta_df['Implemented'] != 0].rename({
            'No.': 'id', 'Name': 'name', 'Subsets': 'subsets', 'Link': 'source_link', 'Description': 'description',
            'HF Link': 'hf_link', 'License': 'license', 'Year': 'year', 'Collection Style': 'collection_style',
            'Language': 'language', 'Dialect': 'dialect', 'Domain': 'domain', 'Form': 'modality', 'Tasks': 'tasks',
            'Volume': 'volume', 'Unit': 'unit', 'Ethical Risks': 'ethical_risk', 'Provider': 'provider',
            'Paper Title': 'paper_title', 'Paper Link': 'paper_link', 'Access': 'access', 'Derived From': 'derived_from', 
            'Test Split': 'is_splitted', 'Notes': 'notes', 'Dataloader': 'dataloader', 'Implemented': 'implemented'
        }, axis=1)
        self._meta_df['is_splitted'] = self._meta_df['is_splitted'].apply(lambda x: True if x =='Yes' else False)

        # Merge Metadata with Config
        name_to_meta_map = {}
        for cfg_meta in self._conhelps:
            # Assign metadata to meta dataframe
            self._meta_df.loc[self._meta_df.dataloader == cfg_meta.dataset_name, [
                'is_large', 'is_resource', 'is_default', 'is_broken',
                'is_local', 'citation', 'license', 'homepage', 'tasks'
            ]] = [
                cfg_meta.is_large, cfg_meta.is_resource, cfg_meta.is_default, cfg_meta.is_broken, 
                cfg_meta.is_local, cfg_meta.citation, cfg_meta.license, cfg_meta.homepage, '|'.join([task.value for task in cfg_meta.tasks])
            ]

            if cfg_meta.dataset_name not in name_to_meta_map:
                name_to_meta_map[cfg_meta.dataset_name] = {}
            if cfg_meta.config.schema not in name_to_meta_map[cfg_meta.dataset_name]:
                name_to_meta_map[cfg_meta.dataset_name][cfg_meta.config.schema] = []
            name_to_meta_map[cfg_meta.dataset_name][cfg_meta.config.schema].append(cfg_meta)
        
        self._meta_df = self._meta_df.fillna(False)
        for dset_name in name_to_meta_map.keys():
            self._meta_df.loc[self._meta_df.dataloader == dset_name, 'metadata'] = MetaDict(data=name_to_meta_map[dset_name])

        if not keep_broken:
            self._meta_df = self._meta_df[~self._meta_df.is_broken]
    
    def filtered(
        self, is_keeper: Callable[[], bool]
    ) -> "SEACrowdMetadataHelper":
        """Return dataset config helpers that match is_keeper."""
        meta_df = self._meta_df[self._meta_df.apply(is_keeper, axis=1, reduce=True)]
        return SEACrowdMetadataHelper(meta_df=meta_df)

    def filter_and_load(
        self, is_keeper: Callable[[], bool]
    ) -> "Dict<str, Dataset>":
        """Return dataset that match is_keeper."""
        filtered_helper = self.filtered(is_keeper)
        for metas in filtered_helper._meta_df.metadata:
            if schema in metas.data:
                for meta in metas.data[schema]:
                    if len(meta.languages) > 1:
                        if lang in meta.config.name:
                            datasets[meta.config.name] = meta.load_dataset()
                    else:
                        datasets[meta.config.name] = meta.load_dataset()
                
    @property
    def available_dataset_names(self) -> List[str]:
        return sorted(self._meta_df.name)

    def __repr__(self):
        return self._meta_df.to_string()

    def __str__(self):
        # return self.__repr__()
        return "test"

    def __iter__(self):
        for row in self._meta_df.iterrows():
            yield row

    def __len__(self):
        return len(self._meta_df)
    
###
# SEACrowd Interface
###

def list_datasets(with_config=False):
    config_helper = SEACrowdConfigHelper()
    return config_helper.list_datasets(with_config=with_config)

def load_dataset(dataset_name, schema='seacrowd'):
    config_helper = SEACrowdConfigHelper()
    return config_helper.load_dataset(dataset_name=dataset_name, schema=schema)

def load_datasets(dataset_names, schema='seacrowd'):
    config_helper = SEACrowdConfigHelper()
    return config_helper.load_datasets(dataset_names=dataset_names, schema=schema)

def load_dataset_by_config_name(config_name):
    config_helper = SEACrowdConfigHelper()
    return config_helper.load_dataset_by_config_name(config_name=config_name)

def load_datasets_by_config_names(config_names):
    config_helper = SEACrowdConfigHelper()
    return config_helper.load_datasets_by_config_names(config_names=config_names)

def list_benchmarks():
    config_helper = SEACrowdConfigHelper()
    return config_helper.list_benchmarks()

def load_benchmark(benchmark_name):
    config_helper = SEACrowdConfigHelper()
    return config_helper.load_benchmark(benchmark_name=benchmark_name)

def for_dataset(dataset_name, schema='seacrowd'):
    config_helper = SEACrowdConfigHelper()
    return config_helper.for_dataset(dataset_name=dataset_name, schema=schema)

def for_datasets(dataset_names, schema='seacrowd'):
    config_helper = SEACrowdConfigHelper()
    return config_helper.for_datasets(dataset_names=dataset_names, schema=schema)

def for_config_name(config_name):
    config_helper = SEACrowdConfigHelper()
    return config_helper.for_config_name(config_name=config_name)

def for_config_names(config_names):
    config_helper = SEACrowdConfigHelper()
    return config_helper.for_config_names(config_names=config_names)

def available_dataset_names(schema=None):
    config_helper = SEACrowdConfigHelper()
    return config_helper.available_dataset_names(schema=schema)

def available_config_names(dataset_name=None):
    config_helper = SEACrowdConfigHelper()
    return config_helper.available_config_names(dataset_name=dataset_name)

if __name__ == "__main__":
    print(f'LIST DATASETS')
    dset_names = list_datasets()
    print(dset_names[:10])
    print()

    print(f'LOAD DATASET `{dset_names[1]}`')
    dset = load_dataset(dset_names[1])
    print(dset)
    print()
    
    print(f'LOAD DATASETS [{dset_names[1:4]}]')
    dsets = load_datasets(dset_names[1:4])
    print(dsets)
    print()
    
    print(f'LIST BENCHMARKS')
    benchmark_names = list_benchmarks()
    print(benchmark_names[:3])
    print()

    print(f'LOAD BENCHMARK `{benchmark_names[0]}`')
    benchmark_dsets = load_benchmark(benchmark_names[0])
    print(benchmark_dsets)
    print()