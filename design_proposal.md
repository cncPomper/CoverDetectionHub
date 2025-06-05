# CoverDetectionHub 2025L continuation - Design Proposal

------------------------

## Autors

***Karol Kasperek***

***Wojciech Bożek***

***Mateusz Węcławski***



## Project description
Aim of the actual term is to expand functionalities of  **CoverDetectionHub** created on previous term by Kamil Szczepanik, Dawid Ruciński, Piotr Kitłowski.
 
Actual state of project contains framework and “hub” for **music cover identification** enabling researchers to compare various CSI methods and run experiments through a user-friendly Gradio interface. 

Purposes for term 25L is to try to develop future challenges from previous implementation and add new functionalities which we will describe later.



## Planned functionalities

- obtaining Da-Tacos dataset and uset to improve results of training and evaluation of models
- improving the [LyriCover](https://github.com/DawidRucinski/LyriCover) model for more sophisticated audio features exctraction; training on a larger subset of SHS100k after improving performance of the model, speed up calculating of actual model
- performing more experiments, similar to "Injected Abracadabra" or others found in the literature for example:
  - The impact of song fragmentation on identification involves evaluating whether algorithms can accurately identify cover versions based on short excerpts of songs ~ ByteCover3: Accurate Cover Song Identification on Short Queries by Xingjian Du et al. (2023)
- Augmentations:
  - Comparing the performance of models trained on the original dataset versus those trained on an augmented version
  - Analyzing the impact of specific augmentation techniques (e.g., pitch shifting, time stretching, noise addition) on individual performance metrics.
- Possible results of augmentations:
  - Increase the diversity of the training data, thereby lowering the risk of the model overfitting
  - Enhancements in metrics such as accuracy, precision, recall, and mean average precision
- model for generating covers from given track (if we have capacity to do that - not included in schedule)


## Technology stack
Main technologies in use:
- **Python**: Our proposed technology stack is based on Python, considering its great capabilities for working with data in an easy way.
- **PyTorch**: Deep learning library.
- **Gradio**: User interface will be implemented in Gradio library, because it is a very convenient tool for a fast prototype building.
- **Numpy**: Library for maths operations.
- **Librosa**: Used for audio loading and some feature extraction (MFCC, spectral centroid) in the simpler comparison methods.
- **venv** (or other tool): For making the project portable in an easy way
- **OpenAI Whisper**: Used by Lyricover to transcribe lyrics and measure similarity in lyric space.

## Project proposed schedule

- **Week 1 (20.02 - 23.02)**:	Create team and discord channel
- **Week 2 (26.02 - 01.03)**:	Topic selection and overview of projects from previous terms
- **Week 3 (04.03 - 08.03)**:	Create entrance design proposal
- **Week 4 (11.03 - 15.03)**:	Design proposal refinement and literature study
- **Week 5 (18.03 - 22.03)**:	Understand of Da-Tacos dataset and use it for testing
- **Week 6 (25.03 - 29.03)**:	Dataset augmentations for better coverage of diffrent scenarios
- **Week 7 (01.04 - 05.04)**:	Improving the LyriCover model and testing
- **Week 8 (08.04 - 12.04)**:	Further improvig of LyriCover and training on larger dataset
- **Week 9-12 (15.04 - 10.05)**:	Implementation and testing of new Cover Generator prototype
- **Week 13 (13.05 - 17.05)**:	Prepare tests for existing code
- **Week 14 (20.05 - 24.05)**:	Evaluation of the results and prepare for final presentation
- **Week 15 (27.05 - 31.05)**:   Final presentation and create recording

*It's only demonstration of schedule. Work during weeks may change in case of troubles or low capacity due to other projects.


-----------------------

## Bibliography

### Cover song detection methods - reused research from previous implementation of CoverDetectionHub

| **Paper**                | **Notes**                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| [CoverHunter: Cover Song Identification with Refined Attention and Alignments](https://arxiv.org/pdf/2306.09025) | Paper proposes new method for CSI task called CoverHunter. It explores features deeper with refined attention and alignments. It has 3 crucial modules: 1) Convolution-augumented transformer; 2) Attention-based time pooling module; 3) A novel training scheme. Authors share their excellent results, beating all existing methods at this time. They test in on benchmarks like DaTacos and SHS100K. In general, they propose a state-of-the-art model, which is definitely one of the current best, which is why it is worth including it in our project. PyTorch implementation of this method can be found in repository [CoverHunter](https://github.com/Liu-Feng-deeplearning/CoverHunter), along with checkpoints of pretrained models. |
| [BYTECOVER: COVER SONG IDENTIFICATION VIA MULTI-LOSS TRAINING](https://arxiv.org/pdf/2010.14022v2) | This paper from 2021 introduces new feature learning method for song cover identification task. It is built on a classical ResNet model with improvements designed for CSI. In a set of experiments, authors demonstrate its effectiveness and efficiency. They evaluate the method on multiple datasets, including DaTacos. In the repository [bytecover](https://github.com/Orfium/bytecover), there is a shared implementation of this method with the best-trained model checkpoints. Thoughts: There is no transformer in a method, which may imply worse results than CoverHunter. |
|[ESSENTIA: AN AUDIO ANALYSIS LIBRARY FOR MUSIC INFORMATION RETRIEVAL](https://www.researchgate.netpublication/256104772_ESSENTIA_an_Audio_Analysis_Library_for_Music_Information_Retrieval) | This paper describes a framework for multiple MIR applications. This tool consists of a number of reconfigurable modules that come in handy for researchers. For our case, an interesting approach is to use the harmonic pitch class profile and the chroma features of audio signals to calculate the similarity of two tracks. This model is very basic and well-known; therefore, it will serve as a reference. The used metric in this model is obtained from a binary cross similarity matrix, which could finally be transferred into a numeric value using a smith-waterman sequence alignment algorithm. **We dedcided to reproduce the experiments for embeddings using MFCC and spectral centroid, but using librosa library.** |
|[THE WORDS REMAIN THE SAME: COVER DETECTION WITH LYRICS TRANSCRIPTION](https://archives.ismir.net/ismir2021/paper/000089.pdf) | Some authors have other applications. In this paper, they proposed another approach, called the *Lyrics-recognition-based system and a classic tonal-based system*. The authors used datasets like Da-Tacos and DALI to detect the cover. Moreover, they used a few fusion models: 1) Lyrics recognition state-of-the-art framework obtained in MIREX 2020 and it uses a model *Time Delay Neural Network* (TDNN) trained using the English tracks of the DALI dataset. In the background music preprocessing step *Singing Voice Separation* (SVS). Moreover, the complete framework they used *Mel-Frequency Cepstral Coefficients* (MFCC) 2) To calculate the similarity between pairs of transcripts - String matching 3) Finally, they used Tonal-based cover detection which is called Re-MOVE and its training of dataset part of Da-Tacos. Another one, that is interesting is more classic HPCP features for cover detection. There has also been proposed joint approach implemented by us, [LyriCover](https://github.com/DawidRucinski/LyriCover). It joins the text retrieval with classic methods. |

### Datasets and benchmarks - reused research from previous implementation of CoverDetectionHub

| **Dataset**                | **Details**                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| Da-TACOS – <br>Dataset for Cover Song Identification<br> and Understanding <br> <li>[Da-TACOS Dataset Paper](https://archives.ismir.net/ismir2019/paper/000038.pdf)</li>  <br> <li>[Da-TACOS GitHub Repository](https://github.com/MTG/da-tacos)</li>  | Two subsets: <br> 1. Benchmark Subset (15,000 songs) <br> 2. Cover Analysis Subset (10,000 songs) <br> <br> <li>Annotations obtained with API from SecondHandSongs.com</li> <br> <li>Features extracted from MP3 audio files encoded at 44.1 kHz sample rate</li> <br> <li>**No audio files included, only pre-extracted features and metadata**</li> <br> <li>7 state-of-the-art CSI algorithms benchmarked on the Benchmark Subset</li> <br> <li>Cover Analysis Subset used to study modifiable musical characteristics  </li>   <br> Thoughts: This dataset has become a classic benchamark for testing CSI systems. Moreover, authors of the paper, along with the  dataset, also provided a  framework for feature extraction and benchmarking - [acoss: Audio Cover Song Suite](https://github.com/furkanyesiler/acoss). 'acoss' includes a standard feature extraction framework with audio features for CSI task and open source implementations of seven CSI algorithms. It was designed to facilitate the future work in this line of research. Although dataset in relatively new (2019), both repositories have not been updated since 5 years ago and considering how rapidly MIR domain develops - 5 years is a lot. That is why our project can be an attempt to create a refreshed and modern version of this framework. It would include state-of-the-art methods with hopefully additional datasets to test them. |
|  <br> [Covers80](http://labrosa.ee.columbia.edu/projects/coversongs/covers80/) |   <li>The dataset contains 80 songs, with 2 different performances of each song by different artists (160 tracks in total).  </li>   <br> <li>All audio files are encoded as 32 kbps MP3 (mono, 16 kHz sampling rate, bandwidth limited to 7 kHz). </li>  <br> Thoughts: We will not use the Covers80 dataset as primary dataset  because it is relatively small and is old (2007). Additionally, the audio files are of low quality (32 kbps, 16 kHz mono).The dataset was assembled somewhat randomly, and it may not provide the diversity or representativeness. However, it has become a CSI systems benchmark, that is why, if we have enough time, we will try to include it in out project. <br> Dataset appeared in a paper [THE 2007 LABROSA COVER SONG DETECTION SYSTEM](http://labrosa.ee.columbia.edu/~dpwe/pubs/EllisC07-covers.pdf). |
| [SHS100K](http://millionsongdataset.com/secondhand/) | <li> Contains metadata and audio features for a large number of songs and their covers. </li> <li> Includes a diverse range of musical genres </li>   <li> Metadata: song title, artist, release year </li><br> Thoughts: **This dataset served us as primary for training purposes** |
| [ZAIKS dataset](https://zaiks.org.pl/) | It's a friendly organization in Poland. The organization will provide a music dataset for testing purposes - these will probably be Polish songs and their famous cover versions. |

### Augmentations and further research - looking for methods used in CSI for further development of the project
| **Paper**                | **Notes**                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| [Training audio transformers for cover song identification](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-023-00297-4#Sec12) | In this paper data augmentations are described in detail, providing an example that we can base our work on. Augmentations conducted by the research team include: randomly changing pitch dimension beetwen 0 to 11 bins, modifing tempo variation, duplication or removal of frames, as well as truncating some parts of the audio signal. |
| [WideResNet with Joint Representation Learning and Data Augmentation for Cover Song Identification](https://www.isca-archive.org/interspeech_2022/hu22f_interspeech.pdf) | In this paper few more augmentation techniques are presented. One of method is implemented by cropping input features with varriant lengths in order to accommodate input feature withf different length, another is based on masking vertical and horizontal bins. Those are called frequency and time masking.  |
|[A SEMI-SUPERVISED DEEP LEARNING APPROACH TO DATASET COLLECTION FOR QUERY-BY-HUMMING TASK](https://archives.ismir.net/ismir2023/paper/000077.pdf) | While this paper focuses on MIR methods used for Query-by-humming task (recognizing music piece by humming its part) and development of dataset designed for that problem, it is treated as a specialization of CSI problem by the authors. As such it provides interesting read that may be beneficial in terms of finding new experiments and refinements for our project. Besides it presents another set of augmentations and experiments that were conducted by reserchers and as such can be treated as yet another inspiration in further development. |
