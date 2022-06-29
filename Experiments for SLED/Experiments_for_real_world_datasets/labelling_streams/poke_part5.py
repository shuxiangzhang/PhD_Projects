"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""
import numpy as np
import os
import time
from collections import OrderedDict
from streams.generators.__init__ import *
from data_structures.attribute_scheme import AttributeScheme
from classifier.__init__ import *
from filters.project_creator import Project
from streams.readers.arff_reader import ARFFReader
from tasks.__init__ import *
from drift_detection.cusum import CUSUM
from drift_detection.ddm import DDM
from drift_detection.eddm import EDDM
from drift_detection.ewma import EWMA
from drift_detection.fhddm import FHDDM
from drift_detection.hddm_a import HDDM_A_test
from drift_detection.hddm_w import HDDM_W_test
from drift_detection.page_hinkley import PH
from drift_detection.rddm import RDDM
from drift_detection.seq_drift2 import SeqDrift2ChangeDetector
from drift_detection.adwin import ADWINChangeDetector

# Initializing detectors
adwin = ADWINChangeDetector()
ddm = DDM()
hddm_w = HDDM_W_test()
cusum = CUSUM()
eddm = EDDM()
rddm = RDDM()
ewma = EWMA()
ph = PH()
hddm_a = HDDM_A_test()
fhddm = FHDDM()
seq_drift2 = SeqDrift2ChangeDetector()

detector = adwin



stream_name = 'poker_hand'
# 1. Creating a project
project = Project("projects/single", "base_detector_circle_abrupt"+ stream_name)

# 2. Generating training stream

project_path = "data_streams/base_detector_circle_abrupt/" + stream_name + "/"
if not os.path.exists(project_path):
    os.makedirs(project_path)


# file_path = project_path
# # Specify the number of drifts
# thresholds = [[[0.2, 0.5], 0.15], [[0.4, 0.5], 0.2], [[0.6, 0.5], 0.25], [[0.8, 0.5], 0.3]]
# stream_generator = CIRCLES(concept_length=5000, transition_length=50, thresholds=thresholds, random_seed=1)
# stream_generator.generate(file_path)
t_1 = time.perf_counter()
# 3. Loading an arff file
file_path = 'C:/Users/szha861/Desktop/Real_world_datasets/'
labels, attributes, stream_records = ARFFReader.read(file_path+"poker-lsn.arff")
attributes_scheme = AttributeScheme.get_scheme(attributes)
located_drifts = [545, 1378, 2563, 3044, 4197, 4550, 5511, 6376, 6665, 7402, 7595, 8492, 8781, 9454, 10095, 10320, 10897, 11186, 11667, 11860, 12341, 12534, 13015, 14136, 14489, 14650, 15515, 16188, 16957, 17822, 18559, 19296, 19553, 20642, 20835, 21124, 22373, 22694, 23943, 24264, 25001, 25322, 26123, 26540, 27725, 27854, 28079, 28976, 29841, 30706, 31027, 31700, 31957, 32630, 33239, 33432, 33817, 34074, 34587, 34844, 35325, 35646, 35903, 36192, 37249, 37538, 38531, 39204, 39717, 39814, 40167, 40552, 40841, 41642, 42603, 43180, 44333, 44718, 45999, 46384, 47185, 47442, 48339, 48532, 49397, 49750, 50647, 51544, 51705, 52314, 52667, 53404, 54045, 54270, 54591, 54816, 55329, 55874, 56099, 56612, 57029, 57190, 57479, 57800, 57961, 58314, 58571, 59628, 60493, 60974, 61935, 62448, 63281, 63730, 64499, 65044, 66325, 66614, 67287, 67736, 67929, 68570, 69659, 70620, 71421, 71614, 72287, 73024, 73729, 74338, 74467, 74948, 76101, 76454, 76999, 77448, 77609, 78186, 78731, 79884, 80397, 80942, 81135, 81712, 82097, 82514, 82739, 83028, 83189, 84054, 84695, 84856, 85369, 85626, 86139, 86428, 87165, 87614, 87871, 88160, 88609, 89730, 90019, 91108, 91429, 92198, 92583, 93160, 93481, 94090, 94347, 95212, 95373, 96206, 96751, 96880, 97777, 98514, 98771, 99316, 99605, 100694, 101399, 102072, 102681, 102938, 103515, 103740, 104221, 104606, 104799, 105088, 105249, 105794, 106083, 106308, 107141, 107398, 108071, 108296, 109161, 109514, 110315, 110444, 111245, 111566, 112399, 112560, 113681, 114546, 115411, 116436, 117109, 117398, 117783, 118264, 118489, 118938, 119131, 119644, 120029, 120446, 121343, 122656, 123521, 123618, 124003, 124388, 125125, 125350, 125671, 126312, 126921, 127306, 128235, 128652, 129133, 129454, 129967, 130704, 131473, 131986, 132275, 132628, 132821, 133750, 134135, 134712, 135321, 136026, 136411, 137596, 137981, 138558, 138815, 139296, 139489, 140226, 140675, 140868, 141157, 141382, 141639, 141864, 142761, 142890, 143211, 143468, 143629, 144174, 144687, 145264, 146193, 146866, 147475, 147828, 148117, 148886, 149143, 149848, 150297, 150618, 151067, 151292, 151837, 152030, 152767, 153088, 154593, 155362, 155619, 156228, 156645, 157062, 157639, 158088, 158249, 158954, 159947, 160652, 161261, 161486, 161903, 162480, 162705, 163154, 163315, 164180, 164885, 165334, 166359, 167320, 167737, 168378, 168827, 169212, 169533, 170142, 170815, 171264, 171617, 172066, 172611, 173188, 173413, 174182, 174663, 174856, 175401, 175818, 177003, 177324, 177773, 178414, 179215, 179632, 180657, 181202, 181811, 182100, 182645, 182870, 183255, 183768, 183993, 184186, 184603, 184796, 185277, 185694, 186815, 187232, 187873, 188674, 189731, 190404, 190725, 191014, 191399, 191592, 191945, 192138, 192203, 192716, 193101, 193678, 194031, 194384, 195089, 195346, 195571, 196628, 197077, 197622, 197847, 198168, 198425, 199226, 199515, 199804, 200797, 201246, 201727, 202048, 203265, 203810, 204099, 204772, 205701, 206982, 207239, 207560, 208329, 208618, 209067, 210060, 210957, 211854, 212143, 213168, 213361, 214226, 214963, 215476, 215605, 215958, 216343, 216760, 217529, 217818, 218619, 220732, 222333, 222558, 223231, 224288, 225025, 225378, 225635, 226084, 226629, 227270, 227943, 228424, 228585, 229770, 231435, 231852, 233293, 233454, 233871, 234160, 234833, 235378, 236211, 237940, 238837, 239574, 239799, 240728, 241049, 241498, 242171, 242524, 243549, 243870, 244863, 245696, 245985, 246786, 247107, 247844, 248101, 248774, 249447, 249736, 250121, 250410, 250539, 251020, 251213, 251694, 252111, 252688, 252913, 253458, 253779, 254068, 254709, 255254, 255799, 256344, 257273, 257626, 258203, 258556, 259421, 259774, 260639, 260992, 262017, 262946, 263907, 264996, 265733, 266310, 266503, 266920, 267177, 267690, 267915, 268364, 268813, 269006, 269423, 270576, 271089, 271410, 271635, 272564, 273173, 274070, 274551, 274904, 275577, 276026, 276603, 277372, 277725, 278942, 279263, 279520, 280097, 280386, 281283, 281540, 282341, 282662, 283591, 284360, 285225, 285994, 286283, 286956, 287597, 288206, 288719, 288944, 289393, 290002, 290291, 290644, 291189, 291478, 292503, 293400, 293913, 294394, 294459, 294940, 295325, 295966, 296319, 296864, 297793, 298178, 299171, 299524, 300421, 300678, 301511, 302536, 303529, 304266, 304523, 305196, 305933, 306350, 306735, 306928, 307409, 307890, 308147, 308532, 309045, 309430, 309623, 310456, 311513, 312890, 313755, 314076, 314397, 314686, 315423, 315584, 316705, 317474, 317731, 318180, 318373, 318854, 319079, 319592, 319817, 320298, 320523, 320940, 321357, 321582, 322383, 322832, 323825, 324850, 325779, 326580, 326997, 327606, 328695, 329400, 329625, 329850, 330267, 330780, 330909, 331454, 331647, 332128, 332641, 333122, 333699, 334052, 334341, 334598, 335271, 335976, 336809, 337098, 337355, 338124, 339085, 339342, 339663, 340240, 341521, 341938, 342387, 343156, 344181, 344886, 345559, 346072, 346905, 347386, 347803, 347996, 348413, 349214, 349471, 349856, 350081, 350306, 351043, 351812, 352133, 352646, 352935, 353288, 353929, 354954, 355883, 356204, 356749, 357614, 358383, 359120, 359697, 359890, 360307, 360820, 361333, 361814, 362231, 362968, 363897, 365178, 365915, 366172, 366717, 366974, 367455, 367680, 368097, 368290, 368579, 369156, 370181, 370982, 371687, 372616, 372745, 373738, 374091, 374348, 374765, 374990, 375471, 375696, 376177, 376370, 376627, 377204, 377365, 377686, 378071, 378456, 378713, 379226, 380411, 381116, 382333, 383678, 384383, 384992, 385089, 385634, 386403, 386852, 387013, 387494, 388647, 389032, 389993, 390826, 391275, 392044, 392365, 392878, 393871, 394384, 394609, 394898, 395187, 395604, 396181, 396694, 397527, 397912, 398265, 398554, 399259, 400540, 401149, 401374, 402431, 402784, 403297, 404098, 405219, 406756, 407141, 408070, 408391, 408936, 409417, 410026, 410571, 411116, 411661, 411854, 412143, 412368, 412977, 413490, 413907, 414356, 414901, 415222, 415895, 416856, 416953, 417658, 417979, 418268, 418589, 419070, 420095, 420544, 420833, 421122, 421475, 421636, 421957, 422758, 423879, 424584, 424873, 425034, 425579, 425868, 426669, 427982, 428271, 428880, 429873, 430642, 430899, 431348, 432213, 432534, 432983, 433432, 434041, 434874, 435291, 436444, 436733, 437470, 438047, 439360, 439873, 441218, 441859, 442724, 443525, 444902, 445191, 445512, 445929, 446122, 447307, 449676, 450125, 450894, 451599, 453040, 453361, 454130, 454707, 455572, 456437, 456726, 457079, 457720, 458169, 458522, 459451, 459836, 460797, 461566, 461791, 462144, 462561, 462850, 463907, 464740, 465029, 465382, 465639, 466184, 466409, 466954, 467179, 467660, 468045, 468430, 468687, 469456, 469809, 470866, 471475, 471796, 472405, 472566, 472727, 473112, 473465, 474362, 474459, 475004, 475421, 475678, 476799, 477120, 478049, 478786, 479267, 479716, 480005, 480678, 480967, 481640, 482185, 482442, 482827, 483116, 483629, 483854, 484335, 484720, 484945, 485938, 486227, 486644, 487701, 488470, 489335, 490104, 490393, 490842, 491419, 491644, 491933, 492702, 492863, 493792, 494561, 494978, 495491, 496420, 497125, 497670, 497895, 498952, 499177, 499658, 500043, 500268, 501005, 501422, 501839, 503024, 503505, 504114, 504563, 504980, 505333, 505814, 506679, 506968, 507865, 508122, 509307, 510492, 510749, 511646, 511871, 512672, 513697, 514498, 515363, 516804, 517253, 517606, 518055, 518888, 519113, 519498, 520043, 520556, 520781, 521038, 521583, 522896, 523601, 524146, 524243, 524532, 525173, 525590, 526327, 527288, 527481, 527866, 528219, 528764, 529021, 529534, 529727, 530208, 530369, 530850, 531363, 531556, 532421, 532710, 532903, 533672, 533929, 534666, 534891, 535372, 536621, 537038, 537359, 537680, 538513, 538802, 539475, 540020, 540213, 540790, 541559, 542072, 542233, 542522, 542939, 543164, 544029, 544318, 545087, 545376, 546017, 546530, 546627, 547812, 548773, 549702, 550471, 550888, 551401, 552298, 552523, 553004, 553165, 553422, 554415, 554800, 555185, 555698, 555923, 557012, 557621, 558518, 558839, 558968, 559641, 560666, 561627, 562524, 563357, 564126, 564863, 565120, 565441, 565666, 566083, 566852, 567109, 567494, 568007, 568744, 569001, 569802, 570955, 571308, 571789, 572046, 572623, 572912, 573617, 574098, 574291, 574548, 575093, 575318, 576151, 576440, 577049, 577818, 578683, 579772, 580125, 580382, 580831, 581088, 581601, 581826, 582275, 582436, 582725, 583142, 583367, 584264, 585289, 585674, 586411, 587340, 587885, 588430, 589007, 589200, 589681, 589938, 590227, 590740, 591413, 591830, 592727, 593624, 593913, 594298, 594619, 595164, 595645, 595902, 596575, 597248, 597505, 597858, 598083, 598500, 599461, 599750, 600103, 600424, 601161, 601450, 601707, 602284, 603469, 604238, 604847, 606032, 606641, 607218, 608211, 609428, 609717, 609974, 611319, 612440, 613433, 614170, 614491, 614844, 615389, 615742, 616735, 616960, 618017, 618082, 618467, 618820, 619205, 619526, 619751, 620392, 620681, 621098, 622251, 622476, 622669, 622958, 623311, 624464, 625009, 625906, 627091, 628244, 629045, 629302, 629975, 630136, 630905, 631162, 631547, 631964, 632733, 633246, 633759, 634080, 635201, 635650, 636291, 638180, 638725, 640038, 640679, 641864, 642377, 642666, 643211, 643724, 643885, 644942, 645487, 647504, 647729, 648242, 648499, 650420, 650869, 651318, 652119, 652344, 652889, 653242, 653883, 654460, 654813, 655742, 656095, 656384, 657121, 658018, 658307, 658436, 659013, 659238, 659559, 660360, 661097, 661290, 661707, 661996, 662413, 662670, 663471, 664400, 664593, 665202, 666099, 666388, 667029, 667606, 668471, 669752, 670681, 670970, 671227, 671644, 671997, 672926, 673407, 674048, 674401, 675234, 675747, 676228, 676549, 677094, 678087, 678536, 678857, 679306, 679595, 680332, 680845, 681134, 681839, 682704, 683793, 684082, 684691, 685236, 685781, 686422, 686839, 687096, 687961, 688122, 688891, 689244, 690173, 691006, 691775, 691968, 692609, 692898, 693539, 693764, 694181, 694406, 694791, 695368, 695593, 696074, 696235, 696556, 697069, 697838, 699183, 699568, 700017, 700114, 700691, 701588, 701845, 702070, 702583, 702904, 703673, 704634, 705115, 705532, 706301, 707070, 707775, 708416, 709025, 710114, 710467, 711748, 713925, 715206, 715527, 716264, 716553, 717578, 718251, 718860, 719149, 719726, 719983, 720688, 721137, 721586, 721811, 722900, 723253, 723574, 724759, 725784, 726809, 727482, 727963, 728700, 729437, 729726, 730399, 731008, 731873, 732386, 732643, 733092, 733477, 733670, 733927, 734120, 734985, 735306, 736075, 736364, 737005, 737550, 738223, 738704, 739153, 739698, 739891, 740500, 740757, 741430, 742007, 742264, 742649, 743258, 743483, 743964, 744125, 744414, 744991, 746016, 746881, 748130, 749091, 749316, 749989, 750790, 751559, 751784, 752137, 752266, 752747, 753804, 754221, 754638, 755407, 755664, 756497, 757682, 758067, 758548, 758869, 759350, 759639, 760344, 760825, 761210, 761499, 761852, 762077, 762910, 764447, 765312, 766401, 766754, 767427, 767652, 768133, 768358, 768903, 769160, 770665, 771242, 771915, 772108, 772557, 773134, 773903, 774320, 775153, 775730, 776211, 776404, 776693, 777078, 777623, 777944, 778233, 778618, 779099, 780444, 780733, 781214, 781823, 781984, 782401, 783298, 783683, 784484, 784741, 784902, 785543, 786824, 787049, 787338, 787627, 788172, 789005, 789294, 789487, 790640, 791409, 791890, 793043, 794580, 795701, 796406, 797879, 798648, 799033, 799290, 799611, 800220, 801245, 801790, 802079, 802528, 803297, 803586, 803875, 804388, 805573, 806214, 807207, 807752, 808265, 809578, 810379, 811148, 811725, 812622, 813199, 813424, 814001, 814482, 815283, 815540, 815957, 817206, 818103, 818616, 819161, 819578, 819995, 820252, 821597, 822334, 822783, 823776]
print(f'Drifts located are: {located_drifts}')
untrained_error = []
true_drifts = []
for i in range(800,1000):
    # 4. Initializing a Learner
    learner = NaiveBayes(labels, attributes_scheme['nominal'])


    # 5. Creating a Prequential Evaluation Process
    prequential = PrequentialDriftEvaluator_real_world(learner, detector, attributes, attributes_scheme,project)
    #result_1 = prequential.run_2(stream_records,1)
    result_2 = prequential.run_1(stream_records,located_drifts,i,1)
    # prequential = PrequentialDriftEvaluator_real_world(learner, detector, attributes, attributes_scheme,project)
    # result_2 = prequential.run_2(stream_records, 1)
    untrained_error.append(result_2[1][-1])
    print(result_2[1][-1])

print(f'Error rates without adjustment are: {untrained_error}')
print(f'Identified true drifts are: {true_drifts}')
