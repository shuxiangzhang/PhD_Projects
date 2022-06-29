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
from drift_detection.sled import Sled

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

d = [ddm, hddm_w, cusum, eddm, rddm, ewma, ph, hddm_a, seq_drift2, fhddm]


# The function to normalize false positives and delay time
#
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
stream_name = "mixed"
project = Project("projects/single", "sled_mixed_gradual_training/" + stream_name)

# 2. Generating training stream

project_path = "data_streams/sled_mixed_gradual_training/" + stream_name + "/"
if not os.path.exists(project_path):
    os.makedirs(project_path)
train_size = 200
file_path = project_path + stream_name
stream_generator = MIXED(concept_length=5000, noise_rate=0.1,transition_length=800, num_drifts=100)
stream_generator.generate(file_path)
#
# 2. Loading an arff file
labels, attributes, stream_records = ARFFReader.read(file_path + ".arff")
attributes_scheme = AttributeScheme.get_scheme(attributes)

# 3. Initializing a Learner
learner = NaiveBayes(labels, attributes_scheme['nominal'])
drift_acceptance_interval = 200
detected_drifts = []
for detector in d:
    print(detector)
    '''The reason why we put this statement inside this for loop is
    because the variable actual_drift_points will be changed
    every time the following statements are executed'''

    actual_drift_points = [i for i in range(5000, 505000, 5000)]
    # 4. Creating a Prequential Evaluation Process
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
concept_length = 1000
true_change = [i for i in range(5000, 5050000, 5000)]
start_point = 0
for t_c in true_change:
    print(t_c)
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
    # print(f'Current weights: {current_weight}')
    if final_weights is None and not (d is None):
        final_weights = list(current_weight)
    else:
        final_weight = learning_rate * np.array(final_weights) + (1 - learning_rate) * current_weight
        final_weight = final_weight / final_weight.sum()
        final_weights = list(final_weight)
    fw.append(final_weights)

F_MCC = []
F_ACC = []
F_RECALL = []
F_PRE = []
F_KAPPA = []
F_TP =[]
F_FP = []
F_DT = []
for final_w in fw:
    repeat_count = 5
    accuracy = []
    kappa = []
    dt = []
    tp = []
    fp = []
    mem_usage = []
    run_time = []
    fn = []
    tn = []
    mcc = []
    mean_pre = []
    mean_re = []
    # Creating a project
    stream_name = "mixed"
    project = Project("projects/single", "sled_mixed_gradual_testing/" + stream_name)
    # Generating training stream

    project_path = "data_streams/sled_mixed_gradual_testing/" + stream_name + "/"
    if not os.path.exists(project_path):
        os.makedirs(project_path)

    for tm in range(repeat_count):
        stream_generator = MIXED(concept_length=5000, noise_rate=0.1, transition_length=800, num_drifts=100)
        file_path = project_path + str(tm)
        stream_generator.generate(file_path)

        # 3. Loading an arff file
        labels, attributes, stream_records = ARFFReader.read(file_path + ".arff")
        attributes_scheme = AttributeScheme.get_scheme(attributes)

        # 4. Initializing a Learner
        learner = NaiveBayes(labels, attributes_scheme['nominal'])

        # 5. Reset base drift detectors
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

        d = [ddm, hddm_w, cusum, eddm, rddm, ewma, ph, hddm_a, seq_drift2, fhddm]

        # 4. Initializing sled method
        detector = Sled(d)
        detector.final_weights = final_w
        actual_drift_points = [i for i in range(5000, 155000, 5000)]
        drift_acceptance_interval = 1000

        # 5. Creating a Prequential Evaluation Process
        prequential = PrequentialDriftEvaluator(learner, detector, attributes, attributes_scheme,
                                                actual_drift_points, drift_acceptance_interval, project)

        result = prequential.run(stream_records, 1)
        accuracy.append(100 - result[0])
        dt.append(result[1])
        fn.append(30 - result[2])
        tn.append(155000 - 30 - result[3])
        tp.append(result[2])
        fp.append(result[3])
        a = result[2] + result[3]
        b = result[2] + (30 - result[2])
        c = (155000 - 30 - result[3]) + result[3]
        d = (155000 - 30 - result[3]) + (30 - result[2])
        MCC = ((result[2] * (155000 - 30 - result[3])) - result[3] * (30 - result[2])) / np.sqrt(a * b * c * d)
        mcc.append(MCC)
        mem_usage.append(result[4])
        run_time.append(result[5])
        kappa.append(result[6])
        precision = result[2] / (result[2] + result[3])
        recall = result[2] / (result[2] + (30 - result[2]))
        mean_pre.append(precision)
        mean_re.append(recall)

    mean_accuracy = np.mean(accuracy)
    std_accuracy = np.std(accuracy)
    mean_kappa = np.mean(kappa)
    std_kappa = np.std(kappa)
    mean_dt = np.mean(dt)
    std_dt = np.std(dt)
    mean_tp = np.mean(tp)
    std_tp = np.std(tp)
    mean_fp = np.mean(fp)
    std_fp = np.std(fp)
    mean_mem_usage = np.mean(mem_usage)
    std_mem_usage = np.std(mem_usage)
    mean_run_time = np.mean(run_time)
    std_run_time = np.std(run_time)
    mean_fn = np.mean(fn)
    std_fn = np.std(fn)
    mean_tn = np.mean(tn)
    std_tn = np.std(tn)
    mean_mcc = np.mean(mcc)
    std_mcc = np.std(mcc)
    std_precision = np.std(mean_pre)
    std_recall = np.std(mean_re)
    m_precision = np.mean(mean_pre)
    m_recall = np.mean(mean_re)
    F_MCC.append([mean_mcc, std_mcc])
    F_ACC.append([mean_accuracy, std_accuracy])
    F_RECALL.append([m_recall, std_recall])
    F_PRE.append([m_precision, std_precision])
    F_KAPPA.append([mean_kappa, std_kappa])
    F_TP.append([mean_tp, std_tp])
    F_FP.append([mean_fp, std_fp])
    F_DT.append([mean_dt, std_dt])
    print(F_MCC)
    print(F_ACC)
    print(F_RECALL)
    print(F_KAPPA)
    print(F_TP)
    print(F_FP)
    print(F_DT)

    f = open('sled_mixed_gradual.txt', 'a')
    f.write(f'\n sled: mean_accuracy {mean_accuracy}, sd_accuracy {std_accuracy}\n')
    f.write(f'\n sled: mean_dt {mean_dt}, sd_dt{std_dt}\n')
    f.write(f'\n sled: mean_tp {mean_tp}, std_tp {std_tp}\n')
    f.write(f'\n sled: mean_fp {mean_fp}, std_fp {std_fp}\n')
    f.write(f'\n sled: mean_mem_usage {mean_mem_usage}, std_mem_usage{std_mem_usage}\n')
    f.write(f'\n sled: mean_run_time {mean_run_time}, std_run_time {std_run_time}\n')
    f.write(f'\n sled: mean_kappa {mean_kappa}, std_kappa{std_kappa}\n')
    f.write(f'\n sled: mean_fn {mean_fn}, std_fn{std_fn}\n')
    f.write(f'\n sled: mean_tn {mean_tn}, std_tn{std_tn}\n')
    f.write(f'\n sled: mean_mcc {mean_mcc}, std_mcc{std_mcc}\n')
    f.write(
        f'sled: \n mean_precision {m_precision}, std_precision {std_precision}\n mean_recall {m_recall}, std_recall {std_recall}')
    f.close()
