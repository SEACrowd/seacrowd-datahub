"""Data Scraping for Alorese Audio-Visual Dataset

Timestamp of Creation: 2024-03-19 (YYYY-MM-DD)

This docstring outlines the process for scraping audio-visual data (WAV and XML files) for the Alorese language from a specific source. The scraping approach involves the following steps:

1. Pagination and BLOB URL Extraction An automated process iterates through paginated content on the source website. The goal is to extract Blob URLs that point to individual detail pages containing the desired data.
2. Detail Page Processing and WAV URL Collection: From each extracted detail page, a mechanism is implemented to target and collect URLs ending with the `.wav` extension. These URLs point directly to the audio files.
3. WAV-XML Matching (Heuristic Approach): Due to potential inconsistencies in file naming conventions between audio (WAV) and caption (XML) files, a heuristic matching strategy is employed. This strategy involves searching within the detail page content (potentially metadata or filenames) for XML files with names that most closely resemble the corresponding WAV filenames.
4. Manual Verification and Handling of Discrepancies:  It is acknowledged that the heuristic matching approach might not always be perfect. Therefore, manual verification of the pairings between WAV and XML files is recommended. Additionally, the process should account for the possibility of WAV files lacking corresponding XML captions.

In summary, this scraping approach automates the acquisition of WAV and XML data from the Alorese language source.
"""


_URLS_DICT = {
    "AOLFM_2016_05_02_frogstory_Basrudin": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_5db33109_854a_47ac_90c9_4e48a6d29643/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_9a4f24cb_9e51_4d29_9b5d_95caecb1f76a/datastream/OBJ/download"
    },
    "AOLFM_2016_05_02_frogstory_Herru": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_0b9c826d_ef06_4242_bc8d_7c7f8f82b5e3/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_c149c856_ae0f_4f6b_82f9_876ab5b69c5b/datastream/OBJ/download"
    },
    "AOLFM_2016_06_17_Aleng_Keleng_Nur": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_608fc96b_a44f_4e9b_a523_e868c0ebf5cf/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_517458bc_9b92_454b_ab24_c692a00cf7e2/datastream/OBJ/download"
    },
    "AOLFM_2016_05_02_frogstory_Pak_Man": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_9b2405de_7ff6_4236_8d8d_2088a4acb474/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_b3746a7d_a221_4cd9_9e07_a868f8b7c6f4/datastream/OBJ/download"
    },
    "AOLFM_2016_05_04_H&F list_Dope": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_80e19ef0_1447_47ac_abe7_d26df37928ac/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_82559327_1e3b_48f4_aeb1_68babcd0bc36/datastream/OBJ/download"
    },
    "AOLFM_2016_05_04_frogstory_Ida": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_e4a0716c_f998_4a28_9244_a579a2ec70b7/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_ed4b9dd2_b2f9_403c_a8a7_c4ab0fca117a/datastream/OBJ/download"
    },
    "AOLFM_2016_05_04_surrey_Tasmin": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_09aef0d1_dd6a_4927_86d7_d09b5b7c42e8/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_8c4706ad_2284_406a_9759_b8a48eed5239/datastream/OBJ/download"
    },
    "AOLFM_2016_05_06_H&F list_Rahma": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_60451deb_68c8_409f_9502_9db844ff397d/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_bc29250b_c625_4c14_ac1b_edfff915f01b/datastream/OBJ/download"
    },
    "AOLFM_2016_05_06_surrey_Dope": { # part2
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_1cdc96ed_6968_4a66_b9f2_68f4a0282db0/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_b51c1b5c_fb66_4721_99a8_c6f97f6ed492/datastream/OBJ/download"
    },
    "AOLFM_2016_05_09_H&F list_Nasrudin": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_400caa1b_176a_4e37_9944_428bc6532dce/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_be740966_8bd1_4ded_baa8_22c5daf5f741/datastream/OBJ/download"
    },
    "AOLFM_2016_05_10_surrey_Herru": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_4fd56462_f432_4082_827c_0dc76775711d/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_ef038455_8bdc_4ce6_a16f_4eabbcbadea7/datastream/OBJ/download"
    },
    "AOLFM_2016_05_11_frogstory_Niati": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_7c83ddd9_1125_4f09_b91d_712b24b86336/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_774ac7dc_a7ff_49c7_8a0c_15c476c5679a/datastream/OBJ/download"
    },
    "AOLFM_2016_05_17_Fato Kada_Niati": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_81cea385_116c_412b_8787_44e202bfb19c/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_553f4351_9b28_42a5_a646_195e141cf748/datastream/OBJ/download"
    },
    "AOLFM_2016_05_17_H&F list_Niati": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_d445bb8e_2223_4aad_b063_b3e7c0010b55/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_169e18f3_0815_4c0d_b8af_65cd59cd627c/datastream/OBJ/download"
    },
    "AOLFM_2016_05_17_H&F list_Nur": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_b50834a6_af60_4e52_b5e1_e5ad58fbfead/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_e9230fd3_5de6_4f7e_8d06_bd9e579ff4ae/datastream/OBJ/download"
    },
    "AOLFM_2016_05_17_surrey_Niati": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_770bca22_2294_4541_a0ae_5184f31b6fbe/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_1817a057_28a8_4515_8af9_f410c59937ec/datastream/OBJ/download"
    },
    "AOLFM_2016_05_17_surrey_Nur": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f0873ac6_63ba_4cf7_ac64_55eb5f805d3d/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_43f1b97e_fba9_4804_b398_c7398e0cf4da/datastream/OBJ/download"
    },
    "AOLFM_2016_05_20_H&F list_Rahma_Tia": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_ed0740e7_80fb_4f52_9b9a_0e050afe1c00/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_12b8644c_e036_45f4_9de3_cfb41d297d13/datastream/OBJ/download"
    },
    "AOLFM_2016_05_20_Ikan_Karo_Rahma_Tia": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_7f6f150a_74ee_4880_8355_5bf64764e62c/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_dd803f9d_74c4_40f1_9f12_2669a184b08e/datastream/OBJ/download"
    },
    "AOLFM_2016_05_20_frogstory_Rahma_Tia": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_bbc1e384_ade3_45b1_bf6f_2708d0eb6227/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_2577ed6d_6f3d_4aa1_b258_f8239a8ef423/datastream/OBJ/download"
    },
    "AOLFM_2016_05_20_surrey_Rahma_Tia": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_702231e5_2077_4fbf_b9ae_76ec7d82690f/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_de4b3dbf_387b_4ccd_8ee7_a8f9659ee2bb/datastream/OBJ/download"
    },
    "AOLFM_2016_05_21_H&F list_Ida_Masang": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_1c27bfd4_2f46_447f_b29a_d06493f2512c/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f9ce6f46_4e2b_498d_a5b7_bc58d6253dbe/datastream/OBJ/download"
    },
    "AOLFM_2016_05_21_Orman_Kotomang_Ida_Masang": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f89585df_36de_452e_b3f7_d1e13927be53/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_3f9d9039_4a79_4fd8_86a6_0b37103b79a7/datastream/OBJ/download"
    },
    "AOLFM_2016_05_21_frogstory_Ida_Masang": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_56c93bb1_5c7b_4863_9f36_3152a4dfdca2/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_22754213_ac44_43a4_897a_94f885979388/datastream/OBJ/download"
    },
    "AOLFM_2016_05_21_surrey_Ida_Masang": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_34fc5477_d6fe_43c0_a0e5_8643730eeadd/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_d6f2d829_a7ad_4452_a868_1074df32a8cf/datastream/OBJ/download"
    },
    "AOLFM_2016_05_22_Anjing_Kera_Rahma_Malihing": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_754b1599_1813_4d2c_8fbe_67ed002c16fd/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_2f5d8ac1_502c_41a4_8ae1_0570545b68b6/datastream/OBJ/download"
    },
    "AOLFM_2016_05_22_H&F list_Ida": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_712d717a_63e6_4956_be55_580596306c86/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_2b4bc8ca_c9c6_453d_b49c_3c5c518e581e/datastream/OBJ/download"
    },
    "AOLFM_2016_05_22_H&F list_Rahma_Malihing": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_a28b4e64_ae52_4a98_a9bc_5fd8edb45996/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_806bfba9_8291_4ced_ad53_8fe53554c24a/datastream/OBJ/download"
    },
    "AOLFM_2016_05_22_frogstory_Rahma_Malihing": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_b46c1d9e_9ab4_416b_94ae_b9ca57427e2c/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_03cbad9c_0f15_4ae3_b93a_aca7a549e9dd/datastream/OBJ/download"
    },
    "AOLFM_2016_05_22_frogstory_Sula": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_d43bbd4f_d18b_43e9_b9c5_4e3d620f0081/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_14fbd734_f837_4deb_983c_8fe9b6391a7d/datastream/OBJ/download"
    },
    "AOLFM_2016_05_22_surrey_Ida": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_b50556d4_5b97_41f5_8836_3f97ee603461/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_e2cbde1e_9c10_4ebc_9528_6f5755fa71f0/datastream/OBJ/download"
    },
    "AOLFM_2016_05_22_surrey_Rahma_Malihing": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_3b56a7ba_c6b9_4b92_9f53_98ad9d9f2b47/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_89f1155e_21b1_451e_b244_4db90ce0fa27/datastream/OBJ/download"
    },
    "AOLFM_2016_05_23_H&F list_Sula": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_52aad4cf_3a40_4f2f_b65b_e5bde6747b36/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_a303d0c2_4ad2_48d4_bcff_e99036ecdae8/datastream/OBJ/download"
    },
    "AOLFM_2016_05_23_Rusa_dan_Kera_Sula": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_2ce71ec4_eba9_4a88_823d_d2c69c50b245/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_270d5f75_e837_4c87_8d0c_7b3146876902/datastream/OBJ/download"
    },
    "AOLFM_2016_05_24_Anjing_dan_Kera_Rahma": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_37b27a59_829b_4c61_9679_a0c15ec37cb6/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_83960176_8417_46a8_9606_f8e3c8374f30/datastream/OBJ/download"
    },
    "AOLFM_2016_05_24_frogstory_Rahma": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_873a8b22_5d83_483e_896a_d1a50d20e7ae/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_5e18d190_ac1e_4704_818f_85fb258df317/datastream/OBJ/download"
    },
    "AOLFM_2016_05_24_surrey_Rahma": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_0b3c225b_18d7_4358_9e69_248a7d6e458d/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_48856dc8_58b3_48c5_a734_a471b703d644/datastream/OBJ/download"
    },
    "AOLFM_2016_05_26_Buaya_dan_Kelinci_Wia": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_051cdce1_4857_4181_bdba_6dc2eb47ec80/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_18ce1bdb_62d6_49ea_a710_b983dd8e8fa5/datastream/OBJ/download"
    },
    "AOLFM_2016_05_26_H&F list_Wia": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_b627f20a_8147_4993_9f44_368edca534a3/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_48131f81_0f6a_4e26_990d_736aac4e6f34/datastream/OBJ/download"
    },
    "AOLFM_2016_05_26_frogstory_Wia": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_309d3437_4a47_4ac7_a1bf_220941a706ac/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_86c9e167_4e39_438f_8723_1d6b42f04071/datastream/OBJ/download"
    },
    "AOLFM_2016_05_26_surrey_Wia": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_9a7c3813_bfdd_45bd_8429_6f62e077978b/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_eeb6e08a_b0c6_4f45_97ad_077cd05b77e0/datastream/OBJ/download"
    },
    "AOLFM_2016_05_27_Fato_Kada_Ina": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_44294f3f_367b_43d0_90eb_dfc14ebe7f56/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_cdb44458_b261_449d_9113_ca718f56f282/datastream/OBJ/download"
    },
    "AOLFM_2016_05_27_H&F list_Ina": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_b358acde_7f37_4ac4_afaf_0780c2237efc/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f9125b5b_c4ef_4828_b76c_854fe6eeeaa6/datastream/OBJ/download"
    },
    "AOLFM_2016_05_27_H&F list_Loni": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_07432ac4_aab6_4bb1_9216_3cb04c2672cd/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_7ee88cd3_a12a_4cf9_a457_ae9c8e8503d0/datastream/OBJ/download"
    },
    "AOLFM_2016_05_27_cerita_Loni": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_3a3fc4d2_801f_4c70_ac16_d845e80d27f0/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_9d82074d_43cb_4ad0_9ea8_72937cbf2d7c/datastream/OBJ/download"
    },
    "AOLFM_2016_05_27_frogstory_Ina": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_5f2b00cb_a9b5_4a68_bf82_73ffdaaa07fa/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_d1141219_592d_401e_95ab_c07eb9f97fee/datastream/OBJ/download"
    },
    "AOLFM_2016_05_27_frogstory_Loni": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_5518f678_7c05_4643_be0d_7afeeaf80e7e/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_53228df4_fee6_45be_9af0_a57c5186ddb7/datastream/OBJ/download"
    },
    "AOLFM_2016_05_27_surrey_Ina": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_3d09e84f_0900_4777_99fb_2d187ac971ac/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_52a439c6_f135_4cb3_8840_cb2044f43f87/datastream/OBJ/download"
    },
    "AOLFM_2016_05_27_surrey_Loni": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_314b24d3_39c2_448c_9cb2_fc56f86fe620/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_62b35ebe_5909_4cf9_9234_41450fc9e88c/datastream/OBJ/download"
    },
    "AOLFM_2016_06_03_cerita-cerita_Nasrudin and Herru": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_4e8f43d2_72c2_431f_a210_a2ed7e57b216/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_a4dafdf5_4636_4bab_978a_a9a66f006d9f/datastream/OBJ/download"
    },
    "AOLFM_2016_06_04_H&F list_Hawa": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_d8bf2b3b_f79d_4523_9915_3304a40cee96/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_6e973ed9_57bd_4255_871e_9f0682bb9d95/datastream/OBJ/download"
    },
    "AOLFM_2016_06_04_H&F list_Ros": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f8a7dc7b_6f48_477c_a0f8_aeec2e289576/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_0913ff89_d279_438b_bea2_e88cdf21b740/datastream/OBJ/download"
    },
    "AOLFM_2016_06_04_cerita-cerita_Hawa": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_0e664e68_c0a3_4cb1_b8a7_7e725f4ec74a/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_0c0f2171_2795_4fb2_87f0_8d0cc73dc810/datastream/OBJ/download"
    },
    "AOLFM_2016_06_04_cerita-cerita_Ros": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_32fe97a6_5162_41d2_8e5e_cc38a4743f9b/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_28f0aa53_19c1_4eca_a985_97ec2306f57c/datastream/OBJ/download"
    },
    "AOLFM_2016_06_04_frogstory_Hawa": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_27b1bdd7_14aa_4d51_b884_d1eccf9f39dd/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_b28d2b20_fc83_4a80_adc0_9ebe6ed36d73/datastream/OBJ/download"
    },
    "AOLFM_2016_06_04_frogstory_Ros": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_b5e6a360_4ff5_4778_8193_29a7e14ad393/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_cb850e11_fa22_4bd9_9636_95587fd30d58/datastream/OBJ/download"
    },
    "AOLFM_2016_06_04_surrey_Hawa": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_c3611384_dd41_41d2_8c1f_21818023da17/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_07aa4bd0_ff35_4dbf_a401_ded79959f796/datastream/OBJ/download"
    },
    "AOLFM_2016_06_04_surrey_Ros": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_95929f61_f662_445a_ac72_d37dd0edbe0b/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_78da9501_64b0_4996_bac2_7ab547b00e09/datastream/OBJ/download"
    },
    "AOLFM_2016_06_11_H&F list_Mona": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_1027bf5b_e814_4706_aefd_bbd505fdf0b7/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_c679b6d8_2d58_42de_95f3_06ae90c77de1/datastream/OBJ/download"
    },
    "AOLFM_2016_06_11_H&F list_Saleha": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_7274b192_7c1c_4235_b4e2_cc565c1c53b2/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_4b487520_e193_4c85_ad18_6051c523d79c/datastream/OBJ/download"
    },
    "AOLFM_2016_06_11_Kotong_Dake_Mona": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_6de6479f_6aac_4279_bc69_06056734a73a/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_88e11326_7de6_4f6b_8f00_d9c34ad4d6d2/datastream/OBJ/download"
    },
    "AOLFM_2016_06_11_cerita-cerita_Saleha": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_46e51792_532e_413c_b1d7_1ecb970d7f02/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_47366acc_2f51_4aef_bde6_27a490c49019/datastream/OBJ/download"
    },
    "AOLFM_2016_06_11_frogstory_Mona": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_ac7fbacd_af58_4bad_a8ff_407ff6ae58d4/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_82a8e51c_4111_4760_9f43_674ea8111b88/datastream/OBJ/download"
    },
    "AOLFM_2016_06_11_frogstory_Saleha": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_fdcaeff1_a0b9_46c5_8a9a_53c70308927d/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_21b82a86_929f_4bdb_952f_7a54852a574d/datastream/OBJ/download"
    },
    "AOLFM_2016_06_11_surrey_Mona": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_a60038ab_6753_4a2e_8fa4_c89ec6b2bc00/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_162b171d_99f0_4532_8b19_a8f79498d732/datastream/OBJ/download"
    },
    "AOLFM_2016_06_11_surrey_Saleha": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_102bf593_e95c_4237_93f7_f1fb3b0f1382/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_de4ee50b_20cb_4eda_84d3_b7184925d985/datastream/OBJ/download"
    },
    "AOLFM_2016_06_12_Gena_Kerani_Intan": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_61231aa6_7b9d_4d94_8bef_1e6c9d1f911e/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_709bc615_cf27_4847_b673_b5c9358485db/datastream/OBJ/download"
    },
    "AOLFM_2016_06_12_H&F list_Intan": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_0154acf2_4aa1_4584_814c_3a5998687bce/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_910bacbb_fd08_4b8c_92df_68230e5767a0/datastream/OBJ/download"
    },
    "AOLFM_2016_06_12_H&F list_Isma": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_1ffafab8_9f84_4375_a325_1be17ce67747/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_e118386c_727b_417f_9275_39768a2d4a00/datastream/OBJ/download"
    },
    "AOLFM_2016_06_12_Pasar_Alor_Isma": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_7f78b04d_98cd_4c51_8327_0f2bfcf8442c/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_7633a71a_87b7_448d_bd4a_b07fb1123f23/datastream/OBJ/download"
    },
    "AOLFM_2016_06_12_frogstory_Intan": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_87b58939_e70e_4bce_9bc1_d659b6e45b4f/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_5c540434_ac4d_4ef2_bfb9_3bbaf966e95f/datastream/OBJ/download"
    },
    "AOLFM_2016_06_12_frogstory_Isma": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_03c78249_c4dd_4fef_a641_133dff3e2427/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_c6bcecd0_49a3_4efa_b222_24a0ecaeba48/datastream/OBJ/download"
    },
    "AOLFM_2016_06_12_surrey_Intan": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_8d5cce16_2718_4517_b53f_7a4efbd2dadd/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_3f063ac3_ddf2_4245_98b7_a21671398fce/datastream/OBJ/download"
    },
    "AOLFM_2016_06_12_surrey_Isma": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_16814208_e3a5_4492_bcc3_e28a678f33e8/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_4d9db69f_ff2c_42f5_afb3_9ad6c43e2b2d/datastream/OBJ/download"
    },
    "AOLFM_2016_06_17_frogstory_Nur": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_1de6d466_ca5e_45ab_97c8_fd8753e6d391/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_9c3d7c8c_37d5_4c0a_8217_52aa64376a7e/datastream/OBJ/download"
    },
    "AOLFM_2016_06_21_Buaya dan kancil_Yati": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_a822efe8_1519_4729_b664_1f6d665bab3b/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_89f9350e_f2e2_4fcd_b6bd_24c78ee7231a/datastream/OBJ/download"
    },
    "AOLFM_2016_06_21_H&F list_Yati": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_51653cee_adbd_4381_ba41_3979a08eecb6/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_b75f7fda_4d7b_4780_99b1_d3a73adba915/datastream/OBJ/download"
    },
    "AOLFM_2016_06_21_H&F_list_Ade": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_0ce28ec8_dca9_4aa8_8102_40214d27a9e3/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_eb6c0237_99ea_4ebc_b9ed_aba495a6d256/datastream/OBJ/download"
    },
    "AOLFM_2016_06_21_cerita-cerita_Ade": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_b6602b83_84fa_481c_8ba0_4e96c7fa2bd1/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f412037c_ef68_461d_9c3e_9ce6db56f646/datastream/OBJ/download"
    },
    "AOLFM_2016_06_21_frogstory_Ade": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_d5eac7d2_70f7_4ecf_a8b3_9a890c33b8ce/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f85f7edd_597b_4ca6_a1fe_e6990f2d1878/datastream/OBJ/download"
    },
    "AOLFM_2016_06_21_frogstory_Yati": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_a1a8c60f_9638_4fec_96df_d8324ed9fd58/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_fcfb7ea6_b483_461c_86d9_63400cda9f04/datastream/OBJ/download"
    },
    "AOLFM_2016_06_21_surrey_Ade": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f38765bb_614d_4919_8485_6cdcf4be8f91/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_93f06e76_0e3b_45fe_ad5c_76442d7e8df1/datastream/OBJ/download"
    },
    "AOLFM_2016_06_21_surrey_Yati": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_4490b2d1_2ad0_44f4_90b2_e52d71103230/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_830d465a_6ac5_4f50_866c_e25214eace0f/datastream/OBJ/download"
    },
    "AOLFM_2016_07_10_H&F list_Margareta": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_cd3e1869_f337_48a8_b8e0_46a51f869394/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f628807a_baad_4bb3_a47b_a66c1b3e0793/datastream/OBJ/download"
    },
    "AOLFM_2016_07_10_Rusa_dan_Duma_Margareta": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f6502f90_0cf8_49aa_9bca_38aa4d98f99f/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_870dfaa2_965e_4a77_b1a4_604b1a0f1672/datastream/OBJ/download"
    },
    "AOLFM_2016_07_10_frogstory_Margareta": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_505e6bf2_8cb4_4fd3_a9ad_1ca9de6cd3a1/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_8dab1ca9_49b5_4d28_9d6e_150f4a30f47a/datastream/OBJ/download"
    },
    "AOLFM_2016_07_10_surrey_Margareta": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_4430819e_5893_474b_a69e_dfb9f77fbea0/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_92ba070b_9489_4cd1_9754_804660cf713a/datastream/OBJ/download"
    },
    "AOLFM_2016_07_11_H&F list_Johanna": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_ab680c53_5883_4e42_8d66_1a3a1d1bbc13/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_679730c5_febc_4438_836e_9f8e2098848d/datastream/OBJ/download"
    },
    "AOLFM_2016_07_11_H&F list_Lusia": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_a1895e5f_ba83_491b_9663_219db7378adc/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f7ea8692_5112_4bd5_8cf7_8a211bc89134/datastream/OBJ/download"
    },
    "AOLFM_2016_07_11_Kafae_Kalake_Johanna": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_5f58b834_c8bf_46f7_8b5e_9f5a409ce272/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_338375f0_0e27_4ca2_b574_62649188affc/datastream/OBJ/download"
    },
    "AOLFM_2016_07_11_Kafae_Kalake_Lusia": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f3ca6b19_f7ae_4b6a_828e_db27a6b6f4f6/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_9e8470a5_de26_4b5a_a72f_c1ea4958d998/datastream/OBJ/download"
    },
    "AOLFM_2016_07_11_frogstory_Domenikus": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_e2cca82c_8ca0_430e_8a1b_6ae9dc4ee795/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_1c7bbaa5_fc07_4e6a_8c5a_23c7cad2399e/datastream/OBJ/download"
    },
    "AOLFM_2016_07_11_frogstory_Johanna": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_da276e77_0432_43ef_be43_844efa2ff6bd/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_b017f618_bf96_4763_bbb0_accbd5c0bd64/datastream/OBJ/download"
    },
    "AOLFM_2016_07_11_frogstory_Lusia": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_40688f1a_ed75_4707_af09_ac34531c5a34/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_b861a658_9cfd_4f34_b3d1_daa17f3bab1c/datastream/OBJ/download"
    },
    "AOLFM_2016_07_11_surrey_Johanna": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_193b4ef9_d3f6_487b_b40f_46a419e5a4eb/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_142f6e01_ed3e_4a46_8bd5_397f5f7a8839/datastream/OBJ/download"
    },
    "AOLFM_2016_07_11_surrey_Lusia": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_45203056_910b_4ac7_a45f_c2085731bb02/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_455b9f68_35f2_495e_bd7a_e355ac2e5a7c/datastream/OBJ/download"
    },
    "AOLFM_2016_07_13_H&F list_Halifa": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f703d3b9_f01f_4d19_bd5d_b034c11bb9ad/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_9aa5ab6b_bb32_40ec_bc1c_5e0760a6f3d9/datastream/OBJ/download"
    },
    "AOLFM_2016_07_13_H&F list_Mahdia": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_6a2271dc_e468_42f0_bd19_e59e5fead36d/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_fae8b421_d7d6_426f_bc8d_d1e0c7a17b4b/datastream/OBJ/download"
    },
    "AOLFM_2016_07_13_H&F list_Maimuna": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_68043cf3_fc8b_4477_b832_c4a52ca3ad75/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_b5d1d9f5_7f2b_4b66_a122_1472053ad436/datastream/OBJ/download"
    },
    "AOLFM_2016_07_13_H&F list_Marifat": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_abadaf01_d98c_4a46_8f97_61d7b991b5c6/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_a1fd6b17_acf5_4488_86bb_efbda77b393c/datastream/OBJ/download"
    },
    "AOLFM_2016_07_13_Kotong_Dake_Marifat": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_09f035c8_5771_4fac_99c7_dbe8b601bb0b/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_6feaee2e_261d_4045_8969_e2ea90ba5238/datastream/OBJ/download"
    },
    "AOLFM_2016_07_13_Raja_Tanatuka_Maimuna": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_2d9fb603_253f_41ea_aa88_42b8bb4c6120/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_e877e311_d63c_4cbd_a510_4dc4f4c15a3f/datastream/OBJ/download"
    },
    "AOLFM_2016_07_13_Raja_dan_kafae_pito_Halifa": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_7372dc0a_f455_4008_a1a7_6f84b0592e92/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_e602fcda_fe00_406c_8898_6cb80997e7f0/datastream/OBJ/download"
    },
    "AOLFM_2016_07_13_cerita-cerita_Mahdia": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_1d400da7_6bcc_4554_b015_9f110b847aa3/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_2d5e5c56_b5b6_4b81_9fb0_c5b9983cb4f3/datastream/OBJ/download"
    },
    "AOLFM_2016_07_13_frogstory_Halifa": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_7a56ebf9_2ce5_4f31_9a9c_65478332a9e8/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_ec9c4905_5191_424e_a138_6de7d5c4dc46/datastream/OBJ/download"
    },
    "AOLFM_2016_07_13_frogstory_Mahdia": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_90eccc00_2606_4558_ada9_3ec1ced07791/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f59f5777_52f0_4a69_90cc_061cfcb1ea62/datastream/OBJ/download"
    },
    "AOLFM_2016_07_13_frogstory_Maimuna": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_fca1e728_79fd_48e5_b6c6_fd8275e7a763/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_09e05a88_d928_4c3a_8c61_dbb6c323be03/datastream/OBJ/download"
    },
    "AOLFM_2016_07_13_frogstory_Marifat": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_2894ce68_da43_4643_942d_8f9d291a83d9/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_c8d0d661_a586_4e82_8a7b_65c4991e08f0/datastream/OBJ/download"
    },
    "AOLFM_2016_07_13_surrey_Halifa": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_17ba9901_0aa1_4484_a38a_03b218f2c033/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_3f409a55_1031_4274_a673_969758f77bbd/datastream/OBJ/download"
    },
    "AOLFM_2016_07_13_surrey_Mahdia": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_d2431054_f35e_400b_8425_657befb74949/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_a45cb396_3839_44ed_87b5_2ec9d403bb28/datastream/OBJ/download"
    },
    "AOLFM_2016_07_13_surrey_Maimuna": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_4ec90442_c078_410f_a069_ef81cbf5557c/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_5e11d28d_e44b_4ecf_93ba_8deaa3bc4ed8/datastream/OBJ/download"
    },
    "AOLFM_2016_07_13_surrey_Marifat": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_e3a993c4_4fff_4f64_8eef_f71526132cde/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_28c0f332_afae_4bf8_bcee_89a1bc648962/datastream/OBJ/download"
    },
    "AOLFM_2016_07_20_Aho_nang_leki_list_Loriana": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_cd2b11f4_8b34_4ec1_989b_a8ea831c6748/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_8d278176_8ff5_4577_b118_08bf77651da5/datastream/OBJ/download"
    },
    "AOLFM_2016_07_20_Berburu_Onaria": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_28c6bd35_30fd_4519_bab4_b70dbbe2950f/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_75a07f66_b56d_4f4f_832c_3daf7e43077e/datastream/OBJ/download"
    },
    "AOLFM_2016_07_20_H&F list_Halena": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_3c08c56f_1283_4d2f_8864_960312196b8f/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_0861927f_a3a7_4ddc_a5d8_5af0f07c157a/datastream/OBJ/download"
    },
    "AOLFM_2016_07_20_H&F list_Loriana": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_95a288ae_195d_4407_8bf6_04ff5924f75b/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_a2ee2f60_2a9d_4e15_83a3_0022bdbd8682/datastream/OBJ/download"
    },
    "AOLFM_2016_07_20_H&F list_Onaria": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_5108e898_6d12_42d3_8a6f_f0a506d356bc/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_7942a015_4aad_4f56_ad22_6480a733eef6/datastream/OBJ/download"
    },
    "AOLFM_2016_07_20_cerita-cerita_Halena": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_47aae616_32c6_4d3c_80cb_a61c13102916/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_d7d0ccd4_da28_4953_859f_4cd7c7f7a2da/datastream/OBJ/download"
    },
    "AOLFM_2016_07_20_frogstory_Halena": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_711478e6_f3a3_41e3_b0de_e5e3da243e88/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_e45fec6c_396d_4e2a_a50a_9a1ed4705923/datastream/OBJ/download"
    },
    "AOLFM_2016_07_20_frogstory_Loriana": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_6af003c1_398d_493e_ac0a_20cb2aa647a4/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_6b32f4bf_a8c3_4bf7_b23f_a53eaf310eb0/datastream/OBJ/download"
    },
    "AOLFM_2016_07_20_frogstory_Onaria": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_c9aeaede_346b_4f7c_89c5_bb4ec2ab10a1/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_d48a57ad_3bbe_427c_bd3a_761143d72d7e/datastream/OBJ/download"
    },
    "AOLFM_2016_07_20_surrey_Halena": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_1660b8bb_479d_4e61_92bc_84d10299e866/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_0eae5ca3_2957_4823_9b81_6077ee7776f2/datastream/OBJ/download"
    },
    "AOLFM_2016_07_20_surrey_Loriana": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_e551a3b4_ddba_4831_af7c_2a855aa60450/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_cb1f457a_99d4_49af_a619_e8ce7e4a026e/datastream/OBJ/download"
    },
    "AOLFM_2016_07_20_surrey_Onaria": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_4615f78f_57db_422d_9da0_196b14de81d8/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_10d8d094_74ad_48fd_89e7_e57aa47c7fad/datastream/OBJ/download"
    },
    "AOLFM_2016_07_21_H&F list_Agerina": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_aac17be8_30a9_43a8_b70b_5de3ed27254b/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_d6bde0a8_12c2_40cb_8677_4066e84b561e/datastream/OBJ/download"
    },
    "AOLFM_2016_07_21_H&F list_Ani": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f75febea_948c_4c41_a2ba_26ba837a929f/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_7f61588e_df68_49c9_871a_a12ec1bc0f36/datastream/OBJ/download"
    },
    "AOLFM_2016_07_21_H&F list_Mehelina": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_ada60346_e561_47d6_ad17_3c478f08b390/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_863b4caf_1f45_40a5_9d59_e9ae562ca02d/datastream/OBJ/download"
    },
    "AOLFM_2016_07_21_H_F_list_Marta": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_50cd6c4a_271e_467c_8a82_33e8f15ef17d/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_37f96a3a_f16c_4f4d_a819_99aad6c89a8c/datastream/OBJ/download"
    },
    "AOLFM_2016_07_21_Kasih_makan_babi_Mehelina": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_80464d4d_b2d4_42c8_8027_27c233382a47/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_ebc32c66_e37d_46ed_b340_dccb19e3a05f/datastream/OBJ/download"
    },
    "AOLFM_2016_07_21_cerita-cerita_Agerina": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_717b030c_a5e0_4347_b669_95ef716f5a9a/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_0736e263_8b50_4a17_b328_f5bf03f91a86/datastream/OBJ/download"
    },
    "AOLFM_2016_07_21_cerita-cerita_Ani": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_fdfd8fba_3ca8_4669_8b40_65aaaa1109c1/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_fcf572cc_ce52_4997_a073_f3af8fbcfdb0/datastream/OBJ/download"
    },
    "AOLFM_2016_07_21_cerita-cerita_Marta": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_ed7bbe68_013a_4b40_ba6b_d69fd40f7e3b/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_83ee48e8_6f21_49e5_996a_3d6d33afbbdf/datastream/OBJ/download"
    },
    "AOLFM_2016_07_21_frogstory_Agerina": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_ec6b69c2_5a03_49fe_a23f_7adc2e5e8ad3/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_d536d92a_a2f1_4a0d_84a9_4eb34349d490/datastream/OBJ/download"
    },
    "AOLFM_2016_07_21_frogstory_Ani": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_63a28cac_41c8_41de_82b6_7fad1ae52814/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_bb38ddf9_ceba_4782_b0d7_5a8a3d615d9b/datastream/OBJ/download"
    },
    "AOLFM_2016_07_21_frogstory_Marta": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_04660226_c580_42c1_b314_56e4563b37ce/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_40c5460c_0b01_47bc_888e_7064ef786a6c/datastream/OBJ/download"
    },
    "AOLFM_2016_07_21_frogstory_Mehelina": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_a6bdb090_3518_4b58_8593_8460207ca693/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_1f5ae42b_bb30_4357_98d6_e0302eb1f7c3/datastream/OBJ/download"
    },
    "AOLFM_2016_07_21_surrey_Agerina": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_9b2261ba_af25_4caf_a229_8c246a176295/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_783d2bf3_8b7e_42da_aab3_5048ae3afa21/datastream/OBJ/download"
    },
    "AOLFM_2016_07_21_surrey_Ani": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_630a288f_e96b_44f3_a828_161dc81665c2/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_9dfade61_ca66_4f34_adfd_ebfeca44d7cc/datastream/OBJ/download"
    },
    "AOLFM_2016_07_21_surrey_Marta": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f8c0e693_ec1b_4f68_ac9e_1f8f3d50a5df/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_7ace4b4d_0a05_45f7_b6cf_02940d23e235/datastream/OBJ/download"
    },
    "AOLFM_2016_07_21_surrey_Mehelina": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_802caaba_ab0a_44f5_be0c_497f5687bcb0/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_7acd2e86_158f_499b_9f8e_9e1669a4f3bc/datastream/OBJ/download"
    },
    "AOLFM_2016_07_22_H&F list_Josefina": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_ba4a3d1c_beb9_4b94_9be9_9be0fe0ab283/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_d8721ee0_6d9b_4364_86b1_69db2089cfbf/datastream/OBJ/download"
    },
    "AOLFM_2016_07_22_H&F list_Magdalena": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f22099e5_4533_486d_b4f8_52fd0687c0ce/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_783631d1_153d_4812_b392_d4704f5d05f2/datastream/OBJ/download"
    },
    "AOLFM_2016_07_22_H&F list_Matilda": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_7e14119c_86b7_49f3_8fe4_79cf4a41c86b/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_6c9e1d12_e4f5_433b_9e80_bcf3f03fd3bd/datastream/OBJ/download"
    },
    "AOLFM_2016_07_22_Potong_sayur_Josefina": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_224f4920_0ee1_4cb9_9f60_afa43c10e264/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_14ec02a1_49a1_4595_a14a_41d998bb20f9/datastream/OBJ/download"
    },
    "AOLFM_2016_07_22_cerita-cerita_Magdalena": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_e39e3cbe_f5d1_4d3b_a587_7b764a9040b8/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_5123d677_c418_490c_8f16_097285b9dbc4/datastream/OBJ/download"
    },
    "AOLFM_2016_07_22_cerita_agar-agar_Matilda": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_3f1ea671_2819_45e2_a74a_f0ad31eb5d91/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_aa922b7b_9d9d_4739_af8c_07767bda5399/datastream/OBJ/download"
    },
    "AOLFM_2016_07_22_frogstory_Josefina": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_0e04a727_e994_4764_991f_0929534da3c7/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_76ee9a30_118e_4926_a54b_96f38fedb680/datastream/OBJ/download"
    },
    "AOLFM_2016_07_22_frogstory_Magdalena": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_801a491a_1355_4a4b_8976_7fb75d0c9857/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_821e7962_1f4c_49fb_9d39_eae3a67b6843/datastream/OBJ/download"
    },
    "AOLFM_2016_07_22_frogstory_Matilda": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_6cc3591b_c1ee_4774_8fa4_4376b4e8d0b7/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_1661c143_13f1_4c43_8452_4725a46d3eb2/datastream/OBJ/download"
    },
    "AOLFM_2016_07_22_surrey_Josefina": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_7c182707_c164_4862_bcf6_a3b07274d67e/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_0993d19b_e836_49d2_a171_48ff10dcd089/datastream/OBJ/download"
    },
    "AOLFM_2016_07_22_surrey_Magdalena": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_3b5b60d2_2dce_41c4_916c_1bae698d6bd9/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_86041aaa_ae5e_44bf_ba64_eb6f9c9090e3/datastream/OBJ/download"
    },
    "AOLFM_2016_07_22_surrey_Matilda": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_d8e4c661_94a7_4e0e_878c_cb85b9ef00a9/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_1b8a2652_9046_4ea2_8533_b887084da621/datastream/OBJ/download"
    },
    "AOLFM_2016_08_03_Aleng_Keleng_Jakobus": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_cf5319c9_d902_4dda_85a8_f3f442fe4700/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_938d2d30_1e5b_4b29_ab64_8ce09d4af03a/datastream/OBJ/download"
    },
    "AOLFM_2016_08_03_H&F list_Daud": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_431ef33f_701c_4ded_8d39_2b16e46be930/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_16d8e788_cea1_4e29_9560_ae5e9be87a43/datastream/OBJ/download"
    },
    "AOLFM_2016_08_03_H&F list_Jakobus": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_fbb8b6d4_4933_41bc_86a1_5f88b1155c0c/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_680185db_a06f_4f5e_9c9b_bf1c1aec3000/datastream/OBJ/download"
    },
    "AOLFM_2016_08_03_cerita-cerita_Daud": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_e1528df3_4285_45df_a9f4_f82b5a41117e/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_c17b154f_02f8_4754_9452_d55d5c8b4970/datastream/OBJ/download"
    },
    "AOLFM_2016_08_03_frogstory_Daud": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_e8aed9c7_650a_4b96_8a60_168bf1aac7ac/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_3267ff7a_023d_462f_b79d_5011bf20fe46/datastream/OBJ/download"
    },
    "AOLFM_2016_08_03_frogstory_Jakobus": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_b27ae024_0a79_4a4f_917a_800553e89c36/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_481b57ea_8b81_431f_bf9e_dddf356ab55b/datastream/OBJ/download"
    },
    "AOLFM_2016_08_03_surrey_Daud": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_864fbea8_b2aa_40ca_b551_8240b6509d1a/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_b5efc5de_043c_4512_98ca_baaacf44bb43/datastream/OBJ/download"
    },
    "AOLFM_2016_08_03_surrey_Jakobus": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_93e65d33_fee6_4a5c_b4ee_ed2c057b1b65/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_54f178da_2e17_4c49_9df6_fe57a5f4bad0/datastream/OBJ/download"
    },
    "AOLFM_2016_08_04_H&F list_Arinda": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_380bbcdf_7347_4367_9f20_31560ee3cd45/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_327af27f_3f1c_41d2_a3ce_31f6e772b211/datastream/OBJ/download"
    },
    "AOLFM_2016_08_04_H&F list_Joseba": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_4d85ea11_78f9_4ac5_bf47_76055d5b132c/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_431ed7a2_8774_4802_b88b_20865b6fd77a/datastream/OBJ/download"
    },
    "AOLFM_2016_08_04_H&F list_Lori": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_7a26ab16_5886_4ced_a70a_057c1f3926c3/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_0262684c_b044_470f_b820_23acca1e775e/datastream/OBJ/download"
    },
    "AOLFM_2016_08_04_H&F list_Maria": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_a8c0ea84_e04e_426b_a2d6_bcccf95d4cff/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_c62f522b_5f82_49a2_bacb_bb2bf469210a/datastream/OBJ/download"
    },
    "AOLFM_2016_08_04_Raja_nang_anang_pito_Maria": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_17352198_5c50_4468_b4c6_34a741fe8b6b/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_6a45b9d6_f669_430c_91e3_10485f550079/datastream/OBJ/download"
    },
    "AOLFM_2016_08_04_cerita-cerita_Joseba": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_5a0b53af_2171_4879_a122_48eef37ee272/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_9a5e13e8_c1d9_4620_be7d_947e1a19fab4/datastream/OBJ/download"
    },
    "AOLFM_2016_08_04_cerita-cerita_Lori": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_1989a189_9afa_4208_92cd_08c864dd2795/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_82fbaeb9_78b7_46f8_8bc9_bf94922a1083/datastream/OBJ/download"
    },
    "AOLFM_2016_08_04_cerita_agar-agar_Arinda": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_f9921550_eafc_49f3_96db_03c6094e204a/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_30b9fa0d_2c77_4d56_b7b6_7ae8a46acd75/datastream/OBJ/download"
    },
    "AOLFM_2016_08_04_frogstory_Arinda": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_70c7917a_13ac_42e8_af2b_1d22f863373a/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_73307e19_c552_4638_bdfe_72001830699b/datastream/OBJ/download"
    },
    "AOLFM_2016_08_04_frogstory_Joseba": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_a69ea253_57cb_4214_a25f_2b9ce4ff6a9a/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_4e1a2384_4a10_40e3_99ea_f5dc799d9d22/datastream/OBJ/download"
    },
    "AOLFM_2016_08_04_frogstory_Lori": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_4fcaafd8_f918_4cde_93a5_536a920be638/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_11f846d1_6a6b_40c1_9cae_0c43a9a24d3d/datastream/OBJ/download"
    },
    "AOLFM_2016_08_04_frogstory_Maria": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_19c1c5d1_04fa_443c_bf9c_37179e09ed93/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_dc0d2b48_8a74_4bd3_90f0_4a755d9ca964/datastream/OBJ/download"
    },
    "AOLFM_2016_08_04_surrey_Arinda": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_356c87cf_1ccf_4c63_9141_74fb5ca9df2d/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_a6a53355_4310_48bf_a985_d50bf2142b50/datastream/OBJ/download"
    },
    "AOLFM_2016_08_04_surrey_Joseba": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_dbc2efff_6b75_44a1_9698_3ebebed783ab/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_61f9608e_124c_4b6c_a2db_a520ec1cfbb7/datastream/OBJ/download"
    },
    "AOLFM_2016_08_04_surrey_Lori": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_7ec8bdf8_09f6_4de2_81f9_2bf5877bd762/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_9d1e099e_db52_4d55_86f9_9924b264eac5/datastream/OBJ/download"
    },
    "AOLFM_2016_08_04_surrey_Maria": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_49cc0555_4d5d_4183_8254_7916a1a951ef/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_0d14738a_28a7_44fc_ae0b_678784302d3c/datastream/OBJ/download"
    },
    "AOL_ADAFM_2016_05_08_surrey_Sula": {
        "audio_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_c5cf4cd4_bb3d_4bcc_af27_c66897a03368/datastream/OBJ/download",
        "text_path": "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_170512ad_1703_4ee9_be1e_56997587df2e/datastream/OBJ/download"
    },
}