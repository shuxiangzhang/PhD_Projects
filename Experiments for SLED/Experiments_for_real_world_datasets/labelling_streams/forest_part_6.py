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

stream_name = 'forest_cover_type'
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

# 3. Loading an arff file
file_path = 'C:/Users/szha861/Desktop/Real_world_datasets/'
labels, attributes, stream_records = ARFFReader.read(file_path+"covtypeNorm.arff")
attributes_scheme = AttributeScheme.get_scheme(attributes)
located_drifts = [897, 1378, 2307, 2436, 3493, 4870, 6951, 7080, 7401, 8714, 9355, 10604, 11149, 11694, 12559, 13552, 14481, 15378, 16499, 16884, 17013, 18166, 19223, 19576, 19641, 20922, 22715, 22940, 23197, 23422, 25567, 25696, 25857, 26178, 26755, 27044, 27301, 27462, 27815, 28136, 28457, 28586, 28875, 29196, 29325, 29582, 29935, 30064, 30417, 30706, 31027, 31156, 31413, 31574, 31799, 31928, 32185, 32314, 32571, 32732, 32957, 33118, 33375, 33504, 33761, 33890, 34563, 34820, 35077, 35334, 35495, 35880, 36105, 36394, 36683, 36940, 37293, 37454, 37839, 38256, 38641, 39026, 39411, 39796, 41333, 41462, 41719, 41848, 42105, 42234, 42491, 42620, 42877, 43006, 43263, 43360, 43649, 43746, 43939, 44132, 44325, 44518, 44807, 44904, 45225, 45322, 45579, 45836, 46125, 46382, 46767, 51952, 52305, 52626, 52979, 53460, 53781, 54102, 54455, 54776, 55097, 55418, 55739, 57404, 57597, 57918, 58207, 58752, 59041, 59234, 59427, 60708, 60997, 61158, 61319, 61416, 61737, 61930, 63755, 64140, 64301, 64590, 65199, 66384, 68369, 68658, 68851, 69140, 70453, 70710, 70935, 71256, 71609, 71834, 72155, 72444, 72797, 73150, 73439, 73600, 73857, 74146, 74435, 74756, 75045, 75334, 75655, 75912, 76041, 76234, 76523, 76620, 76813, 76910, 77103, 77296, 77489, 77554, 77843, 78100, 78229, 78678, 78967, 79160, 79385, 79514, 79803, 80092, 80349, 80638, 80959, 81152, 81377, 81666, 81827, 81988, 82085, 82374, 82695, 82856, 82953, 83242, 83403, 83724, 83885, 84110, 84303, 84624, 84913, 85202, 85491, 85684, 85845, 85974, 86135, 86360, 86649, 86906, 87195, 87516, 87709, 87998, 88287, 89440, 90497, 90626, 92739, 93988, 94789, 95046, 95591, 95848, 96137, 96394, 96683, 96972, 97229, 97518, 97807, 98064, 98321, 99186, 100243, 102836, 104213, 104694, 104983, 105304, 105625, 105946, 106267, 106620, 106717, 106942, 107295, 109312, 110689, 111010, 111395, 111748, 112069, 112422, 112775, 113096, 113449, 113770, 114123, 114444, 115757, 116078, 116399, 116592, 117169, 117522, 118675, 119028, 119381, 120310, 120663, 121016, 121369, 123066, 123995, 124252, 124989, 125086, 125471, 125728, 126593, 126914, 127267, 127524, 127909, 128262, 128519, 129896, 130249, 130890, 132075, 132364, 133069, 133166, 133423, 133808, 134161, 134386, 136275, 136532, 136885, 139126, 139383, 139736, 140089, 140442, 140795, 141116, 141853, 142174, 142527, 142880, 143137, 143394, 143555, 143940, 144293, 144646, 144999, 145352, 145673, 145962, 146123, 146348, 146605, 146798, 146991, 147216, 147441, 147634, 147859, 148052, 148213, 148374, 148535, 148664, 148857, 148986, 149915, 150236, 150493, 151486, 153823, 154016, 154785, 155010, 155235, 155844, 156005, 156102, 156231, 156360, 156489, 156970, 157099, 157260, 157517, 157742, 157967, 158672, 158929, 160338, 160531, 160980, 161589, 161814, 162199, 162520, 162905, 163098, 163579, 163740, 163901, 164094, 164735, 164992, 165377, 165506, 165667, 166212, 166725, 167686, 167847, 167944, 168233, 169098, 169387, 169644, 170093, 171790, 175439, 175728, 175985, 177106, 177491, 178388, 179285, 179734, 179831, 180088, 180345, 181818, 181947, 182268, 183901, 184030, 184223, 184320, 184481, 184578, 184963, 185380, 185669, 185926, 186215, 186472, 186729, 186986, 188683, 191020, 193325, 193614, 193999, 194320, 194609, 194930, 195219, 195540, 196789, 197398, 197687, 198552, 200793, 201594, 201851, 202684, 202941, 204126, 204607, 205024, 205281, 205538, 205731, 207780, 208389, 208774, 209959, 210248, 210537, 210730, 210827, 211020, 211853, 212078, 212207, 213392, 213649, 213970, 214259, 214388, 214645, 214902, 215223, 215576, 216185, 216442, 216731, 217020, 217341, 217598, 217887, 219040, 219233, 219362, 219555, 220068, 220421, 220774, 221159, 221544, 221929, 222314, 222731, 222924, 223309, 223758, 224111, 224272, 224689, 226130, 226547, 227060, 227509, 227990, 228471, 228952, 229433, 229914, 230395, 230876, 231453, 232286, 232799, 233280, 233697, 233826, 234211, 234340, 234693, 235142, 235431, 236072, 236553, 236650, 237035, 237260, 237517, 237742, 237999, 238192, 238449, 238674, 239187, 239380, 239605, 239766, 240055, 240312, 240505, 240762, 240987, 241148, 241437, 241694, 241983, 242464, 243137, 243650, 246435, 246948, 247333, 247846, 248071, 248392, 248585, 248746, 248875, 249196, 249325, 249454, 249839, 249936, 250065, 250418, 250547, 250676, 250837, 251318, 251607, 251960, 252249, 252570, 252731, 252892, 253213, 253502, 253823, 253984, 254209, 254402, 254691, 255012, 255365, 255654, 255975, 256552, 257001, 257194, 257547, 258028, 258285, 258670, 259151, 259248, 260209, 260530, 260915, 261172, 261461, 261590, 261975, 262488, 262969, 263322, 263771, 263900, 264221, 264350, 264671, 264832, 265281, 265538, 265955, 266404, 266533, 266822, 267303, 267784, 268265, 268778, 268907, 269260, 269389, 269774, 269903, 270288, 270417, 270770, 270931, 271028, 271285, 271574, 271799, 271928, 272313, 272538, 272635, 272828, 273053, 273214, 273343, 273600, 273697, 273922, 274051, 274212, 274437, 274822, 275047, 275336, 275625, 275882, 276139, 276524, 276781, 277006, 277359, 277520, 277937, 278290, 278515, 278708, 279093, 279254, 279639, 279864, 280249, 280442, 280795, 281020, 281373, 281598, 281951, 282336, 282529, 282754, 283075, 283460, 283653, 283846, 284231, 284424, 284809, 285002, 285387, 285580, 285965, 286158, 286543, 286736, 287121, 287378, 287667, 287860, 288245, 288502, 288791, 289048, 289273, 289562, 289787, 290044, 290461, 290622, 291135, 291232, 292097, 292162, 292579, 292676, 293125, 293414, 293831, 294088, 294281, 295562, 295755, 295948, 296045, 296238, 296463, 297040, 297361, 297586, 297779, 298004, 298197, 298390, 298583, 298968, 300985, 302490, 303003, 303484, 303965, 304190, 304479, 304960, 305473, 305730, 306211, 306404, 306725, 306918, 307335, 307816, 308201, 308426, 308811, 309196, 309421, 309710, 309903, 310096, 310257, 310482, 310739, 310900, 311029, 311318, 311479, 312024, 312153, 312474, 312731, 313020, 313181, 313278, 313663, 313888, 314241, 314434, 314787, 315012, 315301, 315462, 315687, 316040, 316233, 316458, 316683, 316908, 317101, 317422, 317743, 318064, 318385, 318706, 319027, 319348, 319669, 320022, 320343, 320696, 320985, 321370, 321659, 322044, 322365, 322750, 323039, 323424, 323617, 323874, 324131, 324452, 324837, 325158, 325607, 325992, 326217, 326538, 326923, 327212, 327629, 327918, 328303, 328592, 329009, 329298, 329715, 330004, 330421, 330710, 331127, 331448, 331993, 332154, 332667, 332892, 333373, 333598, 333887, 334304, 334433, 334882, 335107, 335556, 335813, 336262, 336519, 336968, 337225, 337642, 337803, 338188, 338477, 338862, 339023, 339184, 339569, 339826, 339987, 340148, 340373, 340502, 340951, 341144, 341721, 342138, 342363, 342492, 342749, 343038, 343391, 343584, 343873, 344034, 344195, 344356, 344709, 344870, 345319, 345544, 345961, 346154, 346635, 346924, 347213, 347342, 347471, 347856, 348689, 349074, 349203, 349300, 349685, 349814, 350039, 350264, 350681, 350874, 351131, 351484, 351709, 351966, 352095, 352704, 352929, 353186, 353315, 353540, 353829, 353958, 354503, 354856, 354921, 355178, 355787, 356396, 356685, 357006, 357615, 357936, 358129, 358226, 358547, 358868, 359157, 359350, 360055, 360408, 360633, 360986, 361211, 361564, 361789, 362878, 363135, 363328, 363585, 363906, 364131, 364484, 364709, 365062, 365287, 365608, 365833, 366186, 366443, 366636, 366989, 367182, 367663, 368080, 368273, 368754, 369299, 369844, 370197, 370518, 370935, 371192, 371321, 371610, 372059, 372604, 372733, 373086, 373279, 373504, 373665, 373858, 374275, 374468, 374981, 375110, 375463, 375688, 376201, 376426, 376779, 376876, 377101, 377198, 377487, 377680, 377873, 378130, 378323, 378452, 378837, 379030, 379127, 379352, 379481, 379706, 379803, 379996, 380157, 380446, 380671, 380960, 381121, 381378, 381731, 382020, 382213, 382438, 382567, 382952, 383081, 383306, 383627, 383756, 383981, 384270, 384399, 384752, 384977, 385074, 385427, 385620, 385717, 386070, 386295, 386392, 386937, 387162, 387643, 388252, 388477, 388990, 389567, 389632, 390305, 390402, 391043, 391556, 392005, 392166, 392615, 392776, 392937, 393290, 393419, 393932, 394093, 394222, 394767, 394896, 395569, 395986, 396243, 396628, 396789, 396918, 397143, 397464, 397593, 397850, 398139, 398300, 398429, 398686, 398975, 399392, 399649, 399938, 400323, 400612, 400997, 401286, 401671, 401896, 402345, 402666, 403051, 403244, 403565, 404174, 404495, 404848, 405169, 405490, 405715, 405876, 406229, 406518, 406743, 406936, 407225, 407642, 407803, 407996, 408317, 408510, 408671, 408992, 409217, 409474, 409731, 409892, 410053, 410406, 410599, 410728, 410857, 411082, 411563, 411692, 412269, 412398, 412751, 412976, 413105, 413458, 413651, 413780, 413909, 414262, 414839, 415000, 415257, 415418, 415675, 415932, 416093, 416350, 416607, 416768, 417025, 417666, 418275, 418884, 419045, 419462, 419591, 419880, 420041, 420490, 421099, 421516, 421773, 422094, 422319, 422672, 422897, 423218, 423475, 423764, 424021, 424310, 424503, 424856, 425113, 425626, 425787, 425884, 426237, 426686, 428223, 428736, 429217, 429698, 430627, 431076, 431493, 432518, 432743, 432872, 433257, 433642, 434251, 434828, 436877, 437102, 437423, 437840, 438097, 438290, 438483, 438740, 439669, 440950, 441207, 441464, 441785, 442106, 442459, 442780, 443261, 443550, 443903, 444128, 444417, 444706, 444931, 445380, 445509, 445798, 445927, 446792, 447241, 447722, 447979, 449228, 450381, 450478, 450895, 451120, 451345, 451730, 452115, 454388, 454901, 455414, 455575, 455928, 457497, 457818, 458715, 459036, 459389, 460030, 460351, 460736, 461057, 461282, 461539, 461796, 462053, 462278, 462535, 462728, 462889, 463178, 463371, 463468, 463661, 463854, 464175, 464368, 464721, 465266, 465491, 465876, 466229, 466358, 466711, 466968, 467321, 467674, 468251, 468828, 469981, 471518, 474559, 476288, 476577, 476834, 478595, 478884, 479141, 479398, 479943, 480168, 480329, 480682, 481227, 481612, 481805, 481966, 483439, 485104, 485233, 485426, 485619, 485748, 485973, 486262, 486455, 486968, 487193, 487354, 487611, 487964, 488061, 488350, 488447, 488768, 489121, 489218, 489507, 489604, 489925, 490310, 490599, 491016, 491113, 491466, 491563, 491980, 492301, 492718, 492847, 493200, 493297, 493746, 494195, 494356, 494485, 494678, 494807, 495160, 495289, 495674, 495771, 496156, 496285, 496574, 496671, 496800, 497121, 497218, 497347, 497668, 497765, 497894, 498247, 498440, 498793, 499306, 500011, 500492, 501037, 501518, 501615, 502192, 502737, 503282, 503827, 504724, 504917, 505078, 505495, 505720, 505881, 506042, 506299, 506460, 506781, 507038, 507423, 507648, 507905, 508194, 508451, 508708, 508997, 509286, 509543, 509768, 510089, 510378, 510571, 510796, 511085, 511310, 511631, 513200, 513553, 514546, 514803, 515188, 515381, 515670, 515863, 516152, 516345, 516474, 516635, 516860, 516989, 517310, 517471, 517792, 517953, 518274, 518467, 519492, 519909, 520070, 520391, 520520, 520841, 521002, 521291, 521548, 521901, 522254, 522479, 522672, 523121, 523634, 524051, 524564, 524789, 525046, 525239, 525496, 525753, 526010, 526203, 526460, 526685, 526910, 527103, 527392, 527649, 528098, 528355, 528580, 529029, 529254, 529511, 529960, 530153, 530410, 530827, 531244, 531661, 532046, 532271, 532496, 532881, 533266, 533459, 533652, 534005, 534358, 534711, 535032, 535321, 537370, 537627, 538044, 538461, 538718, 539007, 539264, 539553, 539810, 540035, 540292, 540549, 540806, 541575, 541832, 542057, 542250, 542411, 542540, 542765, 543182, 544591, 544720, 544881, 545586, 546387, 546516, 547317, 547798, 548439, 549080, 549177, 549306, 550363, 550556, 550749, 550910, 551103, 551296, 551489, 551682, 551843, 552036, 552229, 552422, 552647, 552840, 553033, 553258, 553451, 553644, 553869, 554062, 554223, 554448, 554609, 554802, 554995, 555188, 555381, 555574, 555767, 555960, 556153, 557914, 558235, 558364, 558589, 558910, 559071, 559136, 559425, 559490, 559619, 559748, 560165, 560454, 560615, 562888, 564009, 565034, 565995, 567116, 569933, 572014, 572783, 576368, 577393, 578322, 578707, 579476, 580725]
print(f'Drifts located are: {located_drifts}')
trained_error = [0.3077, 0.3229, 0.2835, 0.282, 0.274, 0.2517, 0.2802, 0.2816, 0.2819, 0.283, 0.2777, 0.2724, 0.2727, 0.2744, 0.2789, 0.2815, 0.285, 0.284, 0.2764, 0.2748, 0.2743, 0.2642, 0.255, 0.2528, 0.2527, 0.2445, 0.2293, 0.2283, 0.2272, 0.2259, 0.2127, 0.2122, 0.2113, 0.2096, 0.2068, 0.2055, 0.2054, 0.205, 0.2043, 0.2032, 0.2031, 0.2033, 0.2023, 0.202, 0.2022, 0.2016, 0.2014, 0.2019, 0.2013, 0.2004, 0.2003, 0.1999, 0.1998, 0.1998, 0.1996, 0.1994, 0.1995, 0.1992, 0.1995, 0.1995, 0.1996, 0.1996, 0.1991, 0.1992, 0.1986, 0.1984, 0.1968, 0.197, 0.1962, 0.1956, 0.1955, 0.1948, 0.1942, 0.1942, 0.1935, 0.1926, 0.1918, 0.1914, 0.1902, 0.1889, 0.1875, 0.1862, 0.1849, 0.1836, 0.1777, 0.1777, 0.1771, 0.177, 0.1763, 0.1763, 0.1757, 0.1757, 0.1752, 0.1754, 0.1749, 0.1748, 0.1742, 0.1741, 0.1736, 0.1734, 0.1729, 0.1728, 0.1722, 0.1721, 0.1714, 0.1715, 0.1709, 0.1705, 0.1703, 0.1696, 0.1686, 0.1537, 0.1531, 0.1526, 0.1521, 0.1513, 0.1508, 0.1505, 0.15, 0.1498, 0.1494, 0.149, 0.1488, 0.1473, 0.1473, 0.1473, 0.1474, 0.1473, 0.1472, 0.1475, 0.1473, 0.1466, 0.1466, 0.1466, 0.1468, 0.1469, 0.1475, 0.1477, 0.1494, 0.1495, 0.1496, 0.1496, 0.1494, 0.1483, 0.1461, 0.146, 0.1459, 0.1455, 0.1437, 0.1434, 0.1433, 0.1432, 0.143, 0.143, 0.143, 0.1431, 0.1432, 0.1437, 0.144, 0.1439, 0.1441, 0.1443, 0.1446, 0.1446, 0.1448, 0.1449, 0.145, 0.1451, 0.1452, 0.1451, 0.1452, 0.1453, 0.1453, 0.1454, 0.1454, 0.1456, 0.1455, 0.1456, 0.1456, 0.1455, 0.1455, 0.1455, 0.1455, 0.1454, 0.1453, 0.1454, 0.1454, 0.1455, 0.1454, 0.1454, 0.1454, 0.1453, 0.1452, 0.1451, 0.1454, 0.1453, 0.1455, 0.1454, 0.1454, 0.1453, 0.1454, 0.1455, 0.1454, 0.1454, 0.1455, 0.1454, 0.1455, 0.1455, 0.1454, 0.1455, 0.1456, 0.146, 0.1458, 0.146, 0.1459, 0.1459, 0.1459, 0.1461, 0.1462, 0.1463, 0.1466, 0.1467, 0.1466, 0.1456, 0.1452, 0.1452, 0.1436, 0.1422, 0.1416, 0.1414, 0.1412, 0.1411, 0.141, 0.1408, 0.1408, 0.1407, 0.1406, 0.1404, 0.1402, 0.1402, 0.14, 0.1397, 0.1394, 0.1377, 0.137, 0.1367, 0.1366, 0.1364, 0.1362, 0.136, 0.1358, 0.1357, 0.1357, 0.1356, 0.1354, 0.1345, 0.1335, 0.1333, 0.1332, 0.1331, 0.1329, 0.1328, 0.1327, 0.1325, 0.1324, 0.1322, 0.1321, 0.1319, 0.1311, 0.131, 0.1309, 0.1309, 0.131, 0.1309, 0.1309, 0.1309, 0.1312, 0.1316, 0.1317, 0.1319, 0.1321, 0.1332, 0.134, 0.1341, 0.1345, 0.1345, 0.1347, 0.1347, 0.135, 0.1351, 0.1353, 0.1353, 0.1353, 0.1355, 0.1357, 0.1355, 0.1353, 0.1351, 0.1345, 0.1345, 0.1344, 0.1344, 0.1343, 0.1343, 0.1344, 0.1344, 0.1337, 0.1337, 0.1336, 0.1329, 0.1329, 0.1329, 0.1329, 0.133, 0.1331, 0.1332, 0.1333, 0.1333, 0.1333, 0.1333, 0.1332, 0.1333, 0.1333, 0.1333, 0.1334, 0.1335, 0.1336, 0.1337, 0.1338, 0.1339, 0.1341, 0.1341, 0.1342, 0.1344, 0.1344, 0.1344, 0.1346, 0.1345, 0.1345, 0.1347, 0.1346, 0.1347, 0.1347, 0.1347, 0.1347, 0.1347, 0.1344, 0.1345, 0.1345, 0.1344, 0.1335, 0.1335, 0.1333, 0.1334, 0.1334, 0.1332, 0.1332, 0.1332, 0.1332, 0.1333, 0.1332, 0.1332, 0.1333, 0.1333, 0.1333, 0.1334, 0.1334, 0.1334, 0.1335, 0.1336, 0.1337, 0.1339, 0.134, 0.1341, 0.1343, 0.1347, 0.1349, 0.1351, 0.1353, 0.1356, 0.1356, 0.1357, 0.1359, 0.136, 0.1361, 0.1362, 0.1364, 0.1365, 0.1367, 0.1369, 0.1369, 0.137, 0.1371, 0.1372, 0.1373, 0.1374, 0.1374, 0.1374, 0.1385, 0.1387, 0.1389, 0.1393, 0.1395, 0.1397, 0.1397, 0.1397, 0.1397, 0.1398, 0.1399, 0.1397, 0.1397, 0.1397, 0.1396, 0.1395, 0.1397, 0.1397, 0.1397, 0.1398, 0.1399, 0.1401, 0.1403, 0.1404, 0.1406, 0.1408, 0.1409, 0.1411, 0.1411, 0.1419, 0.1424, 0.1425, 0.1426, 0.1427, 0.1428, 0.1429, 0.143, 0.1431, 0.1433, 0.1433, 0.1434, 0.1433, 0.1427, 0.1427, 0.1428, 0.1428, 0.1429, 0.1429, 0.1429, 0.143, 0.143, 0.1431, 0.1431, 0.1425, 0.1423, 0.1423, 0.1421, 0.1421, 0.142, 0.142, 0.142, 0.142, 0.1417, 0.1417, 0.1417, 0.1415, 0.1415, 0.1414, 0.1415, 0.1415, 0.1415, 0.1415, 0.1415, 0.1414, 0.1414, 0.1414, 0.1414, 0.1414, 0.1414, 0.1414, 0.1413, 0.141, 0.141, 0.141, 0.1409, 0.1408, 0.1407, 0.1407, 0.1407, 0.1407, 0.1407, 0.1407, 0.1407, 0.1407, 0.1407, 0.1408, 0.141, 0.1409, 0.1409, 0.1407, 0.1407, 0.1407, 0.1407, 0.1408, 0.1408, 0.1409, 0.141, 0.1411, 0.1413, 0.1414, 0.1415, 0.1414, 0.1415, 0.1416, 0.1417, 0.1417, 0.1418, 0.1419, 0.142, 0.1422, 0.1423, 0.1424, 0.1427, 0.1428, 0.1429, 0.143, 0.143, 0.1431, 0.1431, 0.1432, 0.1432, 0.1433, 0.1434, 0.1435, 0.1436, 0.1436, 0.1437, 0.1437, 0.1438, 0.1439, 0.1439, 0.1439, 0.144, 0.1441, 0.1442, 0.1444, 0.1447, 0.1449, 0.1455, 0.1456, 0.1458, 0.1458, 0.1458, 0.1459, 0.1459, 0.1459, 0.1459, 0.1459, 0.1459, 0.1459, 0.1459, 0.1459, 0.1459, 0.146, 0.146, 0.146, 0.1461, 0.1461, 0.1462, 0.1462, 0.1463, 0.1463, 0.1464, 0.1464, 0.1464, 0.1464, 0.1464, 0.1465, 0.1465, 0.1465, 0.1465, 0.1465, 0.1464, 0.1465, 0.1465, 0.1465, 0.1466, 0.1466, 0.1466, 0.1465, 0.1466, 0.1465, 0.1465, 0.1465, 0.1465, 0.1466, 0.1466, 0.1467, 0.1467, 0.1468, 0.1467, 0.1467, 0.1467, 0.1468, 0.1469, 0.1469, 0.147, 0.147, 0.1471, 0.1472, 0.1473, 0.1474, 0.1474, 0.1474, 0.1474, 0.1474, 0.1474, 0.1475, 0.1475, 0.1475, 0.1476, 0.1476, 0.1476, 0.1477, 0.1477, 0.1478, 0.1478, 0.1479, 0.1479, 0.148, 0.148, 0.148, 0.1481, 0.148, 0.1482, 0.1482, 0.1483, 0.1483, 0.1483, 0.1484, 0.1484, 0.1484, 0.1484, 0.1485, 0.1485, 0.1485, 0.1485, 0.1485, 0.1486, 0.1486, 0.1487, 0.1487, 0.1487, 0.1488, 0.1489, 0.1489, 0.149, 0.149, 0.1491, 0.1492, 0.1492, 0.1492, 0.1493, 0.1492, 0.1492, 0.1493, 0.1493, 0.1493, 0.1493, 0.1493, 0.1493, 0.1493, 0.1494, 0.1495, 0.1495, 0.1495, 0.1495, 0.1496, 0.1496, 0.1496, 0.1496, 0.1496, 0.1496, 0.1496, 0.1496, 0.1496, 0.1496, 0.1496, 0.1497, 0.1496, 0.1497, 0.1497, 0.1498, 0.1498, 0.1498, 0.1499, 0.1499, 0.1499, 0.1499, 0.15, 0.15, 0.1501, 0.1502, 0.1502, 0.1502, 0.1503, 0.1503, 0.1503, 0.1503, 0.1504, 0.1504, 0.1504, 0.1505, 0.1506, 0.1506, 0.1507, 0.1507, 0.1507, 0.1507, 0.1507, 0.1507, 0.1507, 0.1508, 0.1508, 0.1508, 0.1509, 0.1509, 0.1509, 0.1509, 0.1509, 0.1506, 0.1504, 0.1503, 0.1502, 0.1502, 0.1501, 0.1502, 0.1501, 0.1501, 0.1501, 0.1501, 0.1501, 0.1502, 0.1502, 0.1502, 0.1503, 0.1504, 0.1504, 0.1504, 0.1505, 0.1505, 0.1505, 0.1505, 0.1504, 0.1505, 0.1505, 0.1505, 0.1506, 0.1506, 0.1507, 0.1506, 0.1507, 0.1507, 0.1508, 0.1508, 0.1508, 0.1508, 0.1508, 0.1509, 0.1508, 0.1509, 0.1509, 0.151, 0.151, 0.1511, 0.1511, 0.1511, 0.1511, 0.1511, 0.1512, 0.1512, 0.1511, 0.1512, 0.1512, 0.1512, 0.1512, 0.1512, 0.1512, 0.1511, 0.1512, 0.1511, 0.1511, 0.1511, 0.1511, 0.1511, 0.1511, 0.1511, 0.1511, 0.1511, 0.1511, 0.1511, 0.1511, 0.151, 0.1511, 0.1511, 0.1511, 0.1513, 0.1512, 0.1513, 0.1513, 0.1514, 0.1514, 0.1515, 0.1515, 0.1516, 0.1516, 0.1518, 0.1517, 0.1519, 0.1519, 0.152, 0.1519, 0.1521, 0.152, 0.1522, 0.1521, 0.1522, 0.1522, 0.1524, 0.1524, 0.1525, 0.1525, 0.1526, 0.1526, 0.1526, 0.1527, 0.1527, 0.1528, 0.1528, 0.1529, 0.153, 0.1531, 0.1531, 0.1533, 0.1533, 0.1534, 0.1534, 0.1534, 0.1534, 0.1535, 0.1536, 0.1535, 0.1536, 0.1536, 0.1536, 0.1536, 0.1537, 0.1537, 0.1537, 0.1538, 0.1538, 0.1538, 0.1538, 0.1538, 0.1539, 0.1539, 0.1539, 0.1539, 0.1539, 0.1539, 0.154, 0.154, 0.1541, 0.1541, 0.1542, 0.1542, 0.1543, 0.1544, 0.1544, 0.1544, 0.1545, 0.1545, 0.1546, 0.1546, 0.1546, 0.1546, 0.1547, 0.1547, 0.1548, 0.1548, 0.1549, 0.1549, 0.155, 0.1551, 0.1551, 0.1551, 0.1551, 0.1553, 0.1553, 0.1553, 0.1553, 0.1554, 0.1554, 0.1554, 0.1555, 0.1556, 0.1556, 0.1556, 0.1556, 0.1557, 0.1557, 0.1557, 0.1557, 0.1557, 0.1556, 0.1557, 0.1557, 0.1557, 0.1557, 0.1557, 0.1557, 0.1557, 0.1557, 0.1557, 0.1557, 0.1557, 0.1557, 0.1555, 0.1556, 0.1556, 0.1556, 0.1556, 0.1556, 0.1556, 0.1556, 0.1557, 0.1557, 0.1557, 0.1557, 0.1558, 0.1558, 0.1559, 0.156, 0.156, 0.156, 0.1562, 0.1562, 0.1562, 0.1563, 0.1564, 0.1564, 0.1565, 0.1566, 0.1567, 0.1567, 0.1567, 0.1568, 0.1569, 0.1569, 0.1571, 0.1571, 0.1571, 0.1572, 0.1572, 0.1573, 0.1572, 0.1574, 0.1574, 0.1574, 0.1574, 0.1576, 0.1575, 0.1576, 0.1576, 0.1576, 0.1576, 0.1576, 0.1576, 0.1577, 0.1577, 0.1577, 0.1577, 0.1578, 0.1578, 0.1578, 0.1579, 0.1579, 0.1578, 0.1579, 0.1579, 0.1579, 0.1579, 0.1579, 0.158, 0.158, 0.158, 0.158, 0.158, 0.158, 0.158, 0.158, 0.1581, 0.1581, 0.1581, 0.1582, 0.1582, 0.1582, 0.1582, 0.1582, 0.1584, 0.1583, 0.1584, 0.1585, 0.1585, 0.1585, 0.1585, 0.1585, 0.1585, 0.1586, 0.1586, 0.1586, 0.1588, 0.1588, 0.1589, 0.1589, 0.1589, 0.159, 0.1589, 0.159, 0.1591, 0.1591, 0.1591, 0.1592, 0.1591, 0.1592, 0.1592, 0.1592, 0.1593, 0.1593, 0.1593, 0.1594, 0.1594, 0.1595, 0.1596, 0.1596, 0.1596, 0.1596, 0.1596, 0.1597, 0.1597, 0.1597, 0.1597, 0.1597, 0.1597, 0.1597, 0.1597, 0.1598, 0.1598, 0.1598, 0.1599, 0.1599, 0.16, 0.16, 0.16, 0.1601, 0.1601, 0.1601, 0.1602, 0.1603, 0.1602, 0.1603, 0.1604, 0.1604, 0.1604, 0.1605, 0.1605, 0.1605, 0.1605, 0.1605, 0.1605, 0.1605, 0.1605, 0.1605, 0.1605, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1607, 0.1607, 0.1607, 0.1607, 0.1607, 0.1607, 0.1607, 0.1607, 0.1607, 0.1607, 0.1607, 0.1607, 0.1607, 0.1607, 0.1607, 0.1607, 0.1607, 0.1607, 0.1607, 0.1607, 0.1607, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1606, 0.1605, 0.1605, 0.1605, 0.1605, 0.1605, 0.1605, 0.1605, 0.1605, 0.1604, 0.1605, 0.1605, 0.1604, 0.1604, 0.1604, 0.1604, 0.1604, 0.1604, 0.1604, 0.1604, 0.1604, 0.1603, 0.16, 0.16, 0.1599, 0.1598, 0.1596, 0.1595, 0.1595, 0.1593, 0.1593, 0.1593, 0.1593, 0.1593, 0.1592, 0.1592, 0.1589, 0.1589, 0.1589, 0.1589, 0.1589, 0.1589, 0.1589, 0.1589, 0.1589, 0.1588, 0.1588, 0.1588, 0.1589, 0.1589, 0.1589, 0.1589, 0.1589, 0.1589, 0.1589, 0.1589, 0.1589, 0.1589, 0.1589, 0.1588, 0.1588, 0.1588, 0.1588, 0.1588, 0.1587, 0.1587, 0.1587, 0.1586, 0.1585, 0.1585, 0.1585, 0.1585, 0.1585, 0.1585, 0.1585, 0.1583, 0.1583, 0.1583, 0.1583, 0.1583, 0.1582, 0.1582, 0.1581, 0.1581, 0.1581, 0.1581, 0.158, 0.1581, 0.1581, 0.158, 0.158, 0.158, 0.158, 0.158, 0.158, 0.158, 0.1579, 0.158, 0.158, 0.158, 0.158, 0.158, 0.158, 0.158, 0.158, 0.1581, 0.1581, 0.1581, 0.1581, 0.1581, 0.1581, 0.1582, 0.1582, 0.1582, 0.1582, 0.1583, 0.1584, 0.1586, 0.159, 0.1591, 0.1592, 0.1592, 0.1594, 0.1594, 0.1594, 0.1593, 0.1593, 0.1593, 0.1593, 0.1593, 0.1592, 0.1592, 0.1592, 0.1592, 0.159, 0.1588, 0.1588, 0.1588, 0.1588, 0.1588, 0.1588, 0.1588, 0.1588, 0.1588, 0.1587, 0.1587, 0.1587, 0.1588, 0.1588, 0.1588, 0.1588, 0.1588, 0.1589, 0.1589, 0.1589, 0.1589, 0.159, 0.1591, 0.1591, 0.1593, 0.1593, 0.1593, 0.1594, 0.1594, 0.1595, 0.1596, 0.1596, 0.1596, 0.1597, 0.1597, 0.1598, 0.1598, 0.1598, 0.1598, 0.1599, 0.1599, 0.1599, 0.1599, 0.1599, 0.16, 0.16, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1602, 0.1602, 0.1602, 0.1602, 0.1603, 0.1603, 0.1603, 0.1605, 0.1605, 0.1606, 0.1607, 0.1607, 0.1609, 0.1609, 0.161, 0.1611, 0.1612, 0.1612, 0.1612, 0.1612, 0.1613, 0.1612, 0.1612, 0.1612, 0.1612, 0.1612, 0.1612, 0.1613, 0.1613, 0.1613, 0.1613, 0.1613, 0.1613, 0.1613, 0.1613, 0.1613, 0.1613, 0.1613, 0.1614, 0.1613, 0.1613, 0.1614, 0.1614, 0.1614, 0.1613, 0.1614, 0.1615, 0.1615, 0.1616, 0.1615, 0.1616, 0.1616, 0.1616, 0.1615, 0.1615, 0.1616, 0.1615, 0.1616, 0.1615, 0.1615, 0.1615, 0.1615, 0.1615, 0.1615, 0.1614, 0.1614, 0.1614, 0.1614, 0.1614, 0.1614, 0.1614, 0.1614, 0.1614, 0.1614, 0.1614, 0.1614, 0.1614, 0.1614, 0.1614, 0.1614, 0.1614, 0.1613, 0.1613, 0.1613, 0.1613, 0.1612, 0.1612, 0.1612, 0.1611, 0.1611, 0.1611, 0.161, 0.161, 0.1609, 0.1609, 0.1608, 0.1608, 0.1607, 0.1607, 0.1606, 0.1605, 0.1605, 0.1604, 0.1604, 0.1603, 0.1603, 0.1602, 0.1602, 0.1602, 0.1601, 0.16, 0.16, 0.16, 0.1599, 0.1599, 0.1599, 0.1599, 0.1598, 0.16, 0.16, 0.16, 0.16, 0.1599, 0.1599, 0.1599, 0.1599, 0.1599, 0.1598, 0.1598, 0.1598, 0.1598, 0.1596, 0.1596, 0.1596, 0.1596, 0.1596, 0.1596, 0.1596, 0.1595, 0.1593, 0.1593, 0.1593, 0.1592, 0.1591, 0.1591, 0.1589, 0.1588, 0.1587, 0.1586, 0.1586, 0.1586, 0.1584, 0.1584, 0.1584, 0.1584, 0.1584, 0.1583, 0.1583, 0.1583, 0.1583, 0.1583, 0.1583, 0.1583, 0.1583, 0.1584, 0.1583, 0.1584, 0.1584, 0.1584, 0.1584, 0.1584, 0.1584, 0.1584, 0.1584, 0.1584, 0.1584, 0.1584, 0.1584, 0.1583, 0.1583, 0.1583, 0.1583, 0.1579, 0.1579, 0.1579, 0.1579, 0.1579, 0.1579, 0.1579, 0.158, 0.158, 0.158, 0.158, 0.158, 0.158, 0.158, 0.1577, 0.1574, 0.1573, 0.1571, 0.1569, 0.1566, 0.1564, 0.1565, 0.1565, 0.1564, 0.1563, 0.1562, 0.1561, 0.1559]
print(f'Error rates with adjustment are:{trained_error}')
untrained_error = []
true_drifts = []
for i in range(1000,1200):
    # 4. Initializing a Learner
    learner = NaiveBayes(labels, attributes_scheme['nominal'])


    # 5. Creating a Prequential Evaluation Process
    prequential = PrequentialDriftEvaluator_real_world(learner, detector, attributes, attributes_scheme,project)
    #result_1 = prequential.run_2(stream_records,1)
    result_2 = prequential.run_1(stream_records,located_drifts,i,random_seed=1)
    # prequential = PrequentialDriftEvaluator_real_world(learner, detector, attributes, attributes_scheme,project)
    # result_2 = prequential.run_2(stream_records, 1)
    untrained_error.append(result_2[1][-1])
    print(result_2[1][-1])

print(f'Error rates without adjustment are: {untrained_error}')
