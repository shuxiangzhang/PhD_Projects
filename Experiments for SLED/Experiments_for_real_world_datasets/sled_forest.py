"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""
import numpy as np
import os
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
from drift_detection.sled import sled


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

d = [ddm,hddm_w,cusum,eddm,rddm,ewma,ph,hddm_a,seq_drift2,fhddm]

# The function to normalize false positives and delay time

def normalize(item1):
    item1 = np.array(item1)
    if len(item1) == 1:
        return np.array([1])
    else:
        item1 = item1 / item1.sum()
        item1 = 1 - item1
        item1 = item1 / item1.sum()
        return item1

# Training

# 1. Creating a project
stream_name = "forest"
project = Project("projects/single", "sled_cov_training/"+ stream_name)

# 2. Generating training stream

project_path = "data_streams/sled_cov_training/" + stream_name + "/"
if not os.path.exists(project_path):
    os.makedirs(project_path)

file_path = "~/Real_world_datasets/"
# 2. Loading an arff file
labels, attributes, stream_records = ARFFReader.read(file_path+"covTraining.arff")
attributes_scheme = AttributeScheme.get_scheme(attributes)

# 3. Initializing a Learner
learner = NaiveBayes(labels, attributes_scheme['nominal'])
drift_acceptance_interval = 200
detected_drifts = []
for detector in d:

    '''The reason why we put this statement inside this for loop is
    because the variable actual_drift_points will be changed
    every time the following statements are executed'''

    actual_drift_points = [897, 2307, 4870, 6951, 7080, 7401, 8714, 10604, 11149, 11694, 12559, 13552, 16499, 28136, 28457, 29935, 30706, 31156, 31928, 32314, 32732, 33375, 43263, 43649, 43939, 44325, 44807, 45225, 57597, 59041, 61158, 61319, 61416, 61737, 61930, 72797, 73150, 73600, 73857, 74146, 76234, 76523, 76813, 76910, 77103, 77489, 79385, 81666, 83403, 83724, 84110, 84913, 85202, 85491, 85845, 86135, 87516, 116592, 117169, 117522, 119381, 120310, 120663, 121016, 121369, 123066, 123995, 124252, 124989, 125086, 125728, 126914, 127524, 133423, 143137, 143555, 143940, 144293, 144646, 144999, 145352, 145673, 145962, 146348, 146605, 146991, 147216, 147634, 147859, 148052, 148213, 151486, 156005, 156231, 156489, 156970, 157967, 158929, 161589, 161814, 162199, 162520, 162905, 163098, 163579, 163901, 164094, 164735, 165377, 165667, 166212, 166725, 167847, 169644, 171790, 175439, 175728, 175985, 177491, 184030, 184481, 184578, 184963, 185380, 185669, 185926, 186215, 188683, 191020, 193325, 193614, 193999, 194320, 194609, 194930, 195219, 195540, 200793, 204607, 205281, 223758, 224272, 233826, 234211, 234340, 235431, 236072, 236650, 237260, 237517, 237742, 237999, 238192, 238449, 238674, 239187, 239380, 239605, 239766, 240055, 240312, 240505, 240987, 241148, 241437, 241694, 241983, 242464, 243137, 243650, 246948, 248071, 249936, 250065, 250676, 251318, 251960, 252570, 253213, 253502, 253823, 255365, 256552, 260209, 260915, 261461, 263322, 263900, 264350, 264671, 264832, 267303, 268907, 269389, 269774, 269903, 270417, 270770, 270931, 271285, 271574, 271928, 272538, 273053, 273600, 273697, 274437, 274822, 275336, 275882, 276139, 276524, 277006, 277359, 277520, 277937, 278290, 278515, 278708, 279254, 279639, 279864, 280249, 280442, 280795, 281020, 281373, 281598, 281951, 282336, 282529, 282754, 283075, 283460, 283653, 283846, 284231, 284424, 284809, 285002, 285387, 285580, 285965, 286158, 286543, 286736, 287121, 287378, 287667, 287860, 288245, 288502, 288791, 289048, 289273, 289787, 290044, 290622, 292162, 292579, 297040, 297586, 297779, 304190, 306404, 307335, 307816, 308811, 310096, 310739, 311029, 311479, 313278, 313888, 314434, 315012, 315687, 316040, 316233, 316908, 317743, 318385, 319027, 319669, 320022, 320343, 320985, 323617, 323874, 324131, 324452, 325158, 325607, 325992, 326538, 327212, 327629, 327918, 328592, 329298, 330004, 330710, 331448, 332154, 332892, 333373, 333598, 333887, 334304, 334433, 334882, 335107, 335556, 335813, 336262, 336519, 336968, 337225, 337803, 338477, 338862, 339023, 339184, 339826, 340502, 341144, 341721, 342363, 342492, 342749, 343038, 343391, 343584, 344034, 344195, 344356, 344870, 345544, 346154, 346635, 346924, 347342, 347856, 349203, 349300, 349685, 349814, 350264, 350681, 350874, 351131, 351484, 351709, 351966, 352095, 352929, 353186, 353315, 353540, 353829, 353958, 354503, 358129, 364131, 364709, 365608]
    prequential = PrequentialDriftEvaluator(learner, detector, attributes, attributes_scheme,
                                            actual_drift_points, drift_acceptance_interval, project)
    prequential.run(stream_records, 1)
    detected_drifts.append(prequential.located_drift_points)


# 5 Calculatin weights based on the detected drifts
learning_rate = 0.5
fusion_parameter = 0.5
final_weights = None
fw = []
tolerance_length = drift_acceptance_interval
true_change = [897, 2307, 4870, 6951, 7080, 7401, 8714, 10604, 11149, 11694, 12559, 13552, 16499, 28136, 28457, 29935, 30706, 31156, 31928, 32314, 32732, 33375, 43263, 43649, 43939, 44325, 44807, 45225, 57597, 59041, 61158, 61319, 61416, 61737, 61930, 72797, 73150, 73600, 73857, 74146, 76234, 76523, 76813, 76910, 77103, 77489, 79385, 81666, 83403, 83724, 84110, 84913, 85202, 85491, 85845, 86135, 87516, 116592, 117169, 117522, 119381, 120310, 120663, 121016, 121369, 123066, 123995, 124252, 124989, 125086, 125728, 126914, 127524, 133423, 143137, 143555, 143940, 144293, 144646, 144999, 145352, 145673, 145962, 146348, 146605, 146991, 147216, 147634, 147859, 148052, 148213, 151486, 156005, 156231, 156489, 156970, 157967, 158929, 161589, 161814, 162199, 162520, 162905, 163098, 163579, 163901, 164094, 164735, 165377, 165667, 166212, 166725, 167847, 169644, 171790, 175439, 175728, 175985, 177491, 184030, 184481, 184578, 184963, 185380, 185669, 185926, 186215, 188683, 191020, 193325, 193614, 193999, 194320, 194609, 194930, 195219, 195540, 200793, 204607, 205281, 223758, 224272, 233826, 234211, 234340, 235431, 236072, 236650, 237260, 237517, 237742, 237999, 238192, 238449, 238674, 239187, 239380, 239605, 239766, 240055, 240312, 240505, 240987, 241148, 241437, 241694, 241983, 242464, 243137, 243650, 246948, 248071, 249936, 250065, 250676, 251318, 251960, 252570, 253213, 253502, 253823, 255365, 256552, 260209, 260915, 261461, 263322, 263900, 264350, 264671, 264832, 267303, 268907, 269389, 269774, 269903, 270417, 270770, 270931, 271285, 271574, 271928, 272538, 273053, 273600, 273697, 274437, 274822, 275336, 275882, 276139, 276524, 277006, 277359, 277520, 277937, 278290, 278515, 278708, 279254, 279639, 279864, 280249, 280442, 280795, 281020, 281373, 281598, 281951, 282336, 282529, 282754, 283075, 283460, 283653, 283846, 284231, 284424, 284809, 285002, 285387, 285580, 285965, 286158, 286543, 286736, 287121, 287378, 287667, 287860, 288245, 288502, 288791, 289048, 289273, 289787, 290044, 290622, 292162, 292579, 297040, 297586, 297779, 304190, 306404, 307335, 307816, 308811, 310096, 310739, 311029, 311479, 313278, 313888, 314434, 315012, 315687, 316040, 316233, 316908, 317743, 318385, 319027, 319669, 320022, 320343, 320985, 323617, 323874, 324131, 324452, 325158, 325607, 325992, 326538, 327212, 327629, 327918, 328592, 329298, 330004, 330710, 331448, 332154, 332892, 333373, 333598, 333887, 334304, 334433, 334882, 335107, 335556, 335813, 336262, 336519, 336968, 337225, 337803, 338477, 338862, 339023, 339184, 339826, 340502, 341144, 341721, 342363, 342492, 342749, 343038, 343391, 343584, 344034, 344195, 344356, 344870, 345544, 346154, 346635, 346924, 347342, 347856, 349203, 349300, 349685, 349814, 350264, 350681, 350874, 351131, 351484, 351709, 351966, 352095, 352929, 353186, 353315, 353540, 353829, 353958, 354503, 358129, 364131, 364709, 365608]
start_point = 0
for t_c in true_change:
    false_positive = dict()
    delay_time = dict()
    flag = dict()
    for i in range(len(d)):
        false_positive[i] = 0  # Initialize a dictionary for keeping false positives for each base detector
        delay_time[i] = 0  # Initialize a dictionary for keeping delay time
        flag[i] = False
    for detector in range(len(d)):
        for drift in detected_drifts[detector]:
            if drift in range(start_point, t_c + tolerance_length):
                if drift < t_c:
                    false_positive[detector] += 1
                else:
                    if flag[detector] is False:
                        delay_time[detector] = drift - t_c
                        flag[detector] = True
    start_point = t_c + tolerance_length
    normalized_false_positive = []
    normalized_delay_time = []
    sorted_flag = []
    for i in range(len(d)):  # Rearrange the elements in the order of detector id
        normalized_false_positive.append(false_positive[i])
        normalized_delay_time.append(delay_time[i])
        sorted_flag.append(flag[i])
    failed_detector = []
    for i in range(len(d)):
        if not sorted_flag[i]:
            failed_detector.append(i)
    if len(failed_detector) != 0:
        
        # Remove values for failed detectors in both false positives and delay time
        nfp = np.array(normalized_false_positive.copy())
        ndt = np.array((normalized_delay_time.copy()))
        diff = set(list(range(len(nfp)))).difference(set(failed_detector))
        diff = np.array(list(diff))
        if len(diff) != 0:
            nfp = nfp[diff]
            ndt = ndt[diff]
            if all(x == 0 for x in nfp):
                nfp = [1 / len(nfp)] * len(nfp)
            else:
                nfp = list(normalize(nfp))
            if all(x == 0 for x in ndt):
                ndt = [1 / len(ndt)] * len(ndt)
            else:
                ndt = list(normalize(ndt))
            for i in range(len(d)):
                if i in failed_detector:
                    normalized_false_positive[i] = 0
                    normalized_delay_time[i] = 0
                else:
                    normalized_false_positive[i] = nfp.pop(0)
                    normalized_delay_time[i] = ndt.pop(0)
            weight_1 = np.array(normalized_false_positive)
            weight_2 = np.array(normalized_delay_time)
        else:
            weight_1 = np.array([1 / len(nfp)] * len(nfp))
            weight_2 = np.array([1 / len(nfp)] * len(nfp))
    else:
        if all(x == 0 for x in normalized_false_positive):
            weight_1 = np.array([1 / len(normalized_false_positive)]) * len(normalized_false_positive)
        else:
            weight_1 = normalize(normalized_false_positive)
        if all(x == 0 for x in normalized_delay_time):
            weight_2 = np.array([1 / len(normalized_delay_time)]) * len(normalized_delay_time)
        else:
            weight_2 = normalize(normalized_delay_time)

    current_weight = fusion_parameter * weight_1 + (1 - fusion_parameter) * weight_2
    current_weight = current_weight / current_weight.sum()
    if final_weights is None and not (d is None):
        final_weights = list(current_weight)
    else:
        final_weight = learning_rate * np.array(final_weights) + (1 - learning_rate) * current_weight
        final_weight = final_weight / final_weight.sum()
        final_weights = list(final_weight)
    fw.append(final_weights)

stream_name = 'forest'
# 1. Creating a project
project = Project("projects/single", "sled_forest"+ stream_name)

# 2. Generating training stream

project_path = "data_streams/sled_forest/" + stream_name + "/"
if not os.path.exists(project_path):
    os.makedirs(project_path)
actual_drift_points_testing = []
raw_actual_drift_points = [365833, 366443, 366636, 367182, 367663, 370935, 371610, 372733, 373504, 373858, 374468, 375110, 375463, 375688, 376426, 376779, 377198, 377680, 378452, 379127, 379706, 379996, 380446, 380671, 381378, 382020, 382567, 383306, 383981, 384399, 385074, 385717, 386070, 386392, 387162, 387643, 388477, 389632, 390402, 391043, 391556, 392005, 392166, 392776, 393419, 394222, 394896, 396243, 396918, 397850, 398300, 398429, 398686, 398975, 399392, 399649, 399938, 400323, 400997, 401286, 401896, 402345, 402666, 403244, 404495, 404848, 406229, 406518, 406936, 407225, 407642, 408992, 409217, 409731, 410406, 411082, 411692, 423218, 424021, 424503, 424856, 438290, 460351, 461282, 462889, 464368, 464721, 466358, 466711, 469981, 471518, 474559, 476288, 476834, 485748, 487193, 488061, 488350, 488447, 488768, 489121, 489218, 489507, 489604, 489925, 490599, 491466, 491563, 492847, 493200, 493746, 494356, 494485, 494678, 495289, 495674, 495771, 496156, 496285, 496800, 497121, 497218, 497347, 497894, 498247, 499306, 500492, 501037, 501518, 501615, 502737, 505078, 505495, 506042, 506299, 506460, 506781, 507038, 507648, 508997, 509543, 510571, 510796, 513553, 514803, 515381, 516474, 516860, 517310, 524789, 535321, 552422, 552647, 553033, 572014, 572783]
for drift in raw_actual_drift_points:
    actual_drift_points_testing.append(drift-365700)
print(actual_drift_points_testing)

# 3. Loading an arff file
labels, attributes, stream_records = ARFFReader.read(file_path+"covTesting.arff")
attributes_scheme = AttributeScheme.get_scheme(attributes)

# 4. Initializing a Learner
learner = NaiveBayes(labels, attributes_scheme['nominal'])

# 5. Reset base drift detectors

for detector in d:
    detector.reset()

# 4. Initializing sled method

detector = sled(d)
detector.final_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
drift_acceptance_interval = 200

# 5. Creating a Prequential Evaluation Process
prequential = PrequentialDriftEvaluator(learner, detector, attributes, attributes_scheme,
                                        actual_drift_points_testing, drift_acceptance_interval, project)

result = prequential.run(stream_records, 1)
accuracy = (100 - result[0])
dt = (result[1])
fn = (157 - result[2])
tn = (215311 - 157 - result[3])
tp = (result[2])
fp = (result[3])
a = result[2] + result[3]
b = result[2] + (157 - result[2])
c = (215311 - 157 - result[3]) + result[3]
d = (215311 - 157 - result[3]) + (12 - result[2])
MCC = ((result[2] * (215311 - 157 - result[3])) - result[3] * (157 - result[2])) / np.sqrt(a * b * c * d)
mem_usage = result[4]
run_time = result[5]
kappa = result[6]
precision = result[2] / (result[2] + result[3])
recall = result[2] / (result[2] + (157 - result[2]))


f = open('sled_forest','a')
# f.write(f'\n sled: final_weights:{fw[-1]}\n')
f.write(f'\n sled: mean_accuracy {accuracy}')
f.write(f'\n sled: mean_dt {dt}')
f.write(f'\n sled: mean_tp {tp}')
f.write(f'\n sled: mean_fp {fp}')
f.write(f'\n sled: mean_mem_usage {mem_usage}')
f.write(f'\n sled: mean_run_time {run_time}')
f.write(f'\n sled: mean_kappa {kappa}')
f.write(f'\n sled: mean_fn {fn}')
f.write(f'\n sled: mean_tn {tn}')
f.write(f'\n sled: mean_mcc {MCC}')
f.write(f'sled: \n mean_precision {precision}, mean_recall {recall}')
f.close()