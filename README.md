<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- TABLE OF CONTENTS -->

1. [Introduction](#Introduction)
2. [SLED](#SLED)
    1. [Requirements](#Requirements1)
3. [MUDD](#MUDD)
    1. [Requirements](#Requirements2)
4. [ILDD](#ILDD)
    1. [Requirements](#Requirements3)
5. [VAE-DD](#VAE-DD)
    1. [Requirements](#Requirements4)
6. [Contributing](#Contributing)
7. [License](#License)
8. [Contact](#Contact)
9. [Acknowledgments](#Acknowledgments)

<!-- ABOUT THE PROJECT -->
## Introduction <a name="Introduction"></a>

The goal of this project is to detect and interpret drift in evolving data streams. The project includes four concept drift detection methods which are suitable for either supervised or unsupervised learning tasks. These methods are implemented in R or Python.

<p align="right">(<a href="#top">back to top</a>)</p>

## SLED <a name="SLED"></a>

`SLED` is a semi-supervised ensemble drift detector which is designed for supervised learning tasks. It can accurately detect both abrupt and gradual drifts in evolving data streams. The source code for implementing and evaluating the method is included in the folder [Experiments for SLED](https://github.com/shuxiangzhang/PhD_Projects/tree/main/Experiments%20for%20SLED).

### Requirements <a name="Requirements1"></a>

`SLED` is implemented under the [Tornado Framework](https://github.com/alipsgh/tornado). You must have Python 3.5 or above (either 32-bit or 64-bit) installed on your system to run the experiments without any error. For more set-up details, please refer to the [Tornado Framework](https://github.com/alipsgh/tornado).

<p align="right">(<a href="#top">back to top</a>)</p>


## MUDD <a name="MUDD"></a>


`MUDD` is a novel lightweight unsupervised drift detector that is designed for unsupervised learning tasks. It implements a distance-based data transformation mechanism which makes is fast when detecting drifts in multi-dimensional data streams. The source code for implementing and evaluating the method is included in the folder [Experiments for MUDD](https://github.com/shuxiangzhang/PhD_Projects/tree/main/Experiments%20for%20MUDD).


### Requirements <a name="Requirements2"></a>


`MUDD` is implemented in R. To run the experiments, you must have R 4.1.3 or above installed on your system.

**Core dependencies** to be installed:

```bash
install.packages('MASS')
install.packages('dplyr')
install.packages('cramer')
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("GSAR")
install.packages('pryr')
```
<p align="right">(<a href="#top">back to top</a>)</p>

## ILDD <a name="ILDD"></a>

`ILDD` is a novel interpretable unsupervised drift detection framework that can detect concept drifts and interpret concept drifts at the local level. The source code for implementing and evaluating the method is included in the folder [Experiments for ILDD](https://github.com/shuxiangzhang/PhD_Projects/tree/main/Experiments%20for%20ILDD).


### Requirements <a name="Requirements3"></a>

`ILDD` is implemented in R. To run the experiments, you must have R 4.1.3 or above installed on your system.

**Core dependencies** to be installed:

```bash
install.packages('stream')
install.packages('ggpubr')
install.packages('factoextra')
install.packages('philentropy')
install.packages('MASS')
install.packages('dplyr')
install.packages('cramer')
install.packages('LICORS')
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("GSAR")
install.packages('pryr')
```

<p align="right">(<a href="#top">back to top</a>)</p>


## VAE-DD <a name="VAE-DD"></a>


`VAE-DD` is an unsupervised drift detector that is capable of accurately detecting concept drifts in the presence of anomalies. The source code for implementing and evaluating the method is included in the folder [Experiments for ILDD](https://github.com/shuxiangzhang/PhD_Projects/tree/main/Experiments%20for%20VAE-DD).


### Requirements <a name="Requirements4"></a>

`VAE-DD` is implemented in Python. To run the experiments, you must have Python 3.7 or above installed on your system.

**Core dependencies** to be installed:

```bash
pip install numpy
pip install pandas
pip install tensorflow
pip install sklearn
```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing <a name="Contributing"></a>
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. If you have a suggestion that would make this better, please fork the repo and create a pull request.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License <a name="License"></a>

This project is distributed under the MIT License. See [LICENSE.txt](https://github.com/shuxiangzhang/PhD_Projects/blob/main/LICENSE.txt) for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact <a name="Contact"></a>

Shuxiang Zhang - [@LinkedIn](https://www.linkedin.com/in/shuxiang-zhang-523261b7/) - sx.zhang@yahoo.com

Project Link: [https://github.com/shuxiangzhang/PhD_Projects](https://github.com/shuxiangzhang/PhD_Projects)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments <a name="Acknowledgments"></a>

Below is a list of resources I find helpful and would like to give credit to.

* [The Tornado Framework](https://github.com/alipsgh/tornado)
* [Choose an Open Source License](https://choosealicense.com)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/shuxiangzhang/PhD_Projects.svg?style=for-the-badge
[contributors-url]: https://github.com/shuxiangzhang/PhD_Projects/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/shuxiangzhang/PhD_Projects.svg?style=for-the-badge
[forks-url]: https://github.com/shuxiangzhang/PhD_Projects/network/members
[stars-shield]: https://img.shields.io/github/stars/shuxiangzhang/PhD_Projects.svg?style=for-the-badge
[stars-url]: https://github.com/shuxiangzhang/PhD_Projects/stargazers
[issues-shield]: https://img.shields.io/github/issues/shuxiangzhang/PhD_Projects.svg?style=for-the-badge
[issues-url]: https://github.com/shuxiangzhang/PhD_Projects/issues
[license-shield]: https://img.shields.io/github/license/shuxiangzhang/PhD_Projects.svg?style=for-the-badge
[license-url]: https://github.com/shuxiangzhang/PhD_Projects/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/shuxiang-zhang-523261b7/
