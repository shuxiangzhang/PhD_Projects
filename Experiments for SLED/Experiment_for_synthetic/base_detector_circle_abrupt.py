"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""
import numpy as np
import os
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

d = [ddm,hddm_w,cusum,eddm,rddm,ewma,ph,hddm_a,seq_drift2,fhddm]

m_accuracy = OrderedDict()
sd_accuracy = OrderedDict()
m_kappa = OrderedDict()
sd_kappa = OrderedDict()
m_fp = OrderedDict()
m_dt = OrderedDict()
m_tp = OrderedDict()
sd_fp = OrderedDict()
sd_dt = OrderedDict()
sd_tp = OrderedDict()
m_fn = OrderedDict()
sd_fn = OrderedDict()
m_tn = OrderedDict()
sd_tn = OrderedDict()
m_mcc = OrderedDict()
sd_mcc = OrderedDict()
m_run_time = OrderedDict()
sd_run_time = OrderedDict()
m_mem_usage = OrderedDict()
sd_mem_usage = OrderedDict()

stream_name = 'circle'
# 1. Creating a project
project = Project("projects/single", "base_detector_circle_abrupt"+ stream_name)

# 2. Generating training stream

project_path = "data_streams/base_detector_circle_abrupt/" + stream_name + "/"
if not os.path.exists(project_path):
    os.makedirs(project_path)

repeat_count = 30

for detector in d:
    accuracy = []
    kappa = []
    dt = []
    tp = []
    fp = []
    fn = []
    tn = []
    mcc = []
    mem_usage = []
    run_time = []

    for tm in range(repeat_count):
        detector.reset()
        detector.TOTAL_RUNTIME = 0
        file_path = project_path+str(tm)
        # Specify the number of drifts
        thresholds = [[[0.2, 0.5], 0.15], [[0.4, 0.5], 0.2], [[0.6, 0.5], 0.25], [[0.8, 0.5], 0.3]]*7
        thresholds.extend([[[0.2, 0.5], 0.15], [[0.4, 0.5], 0.2], [[0.6, 0.5], 0.25]])
        stream_generator = CIRCLES(concept_length=5000, transition_length=5, thresholds=thresholds, random_seed=tm)
        stream_generator.generate(file_path)

        # 3. Loading an arff file
        labels, attributes, stream_records = ARFFReader.read(file_path+".arff")
        attributes_scheme = AttributeScheme.get_scheme(attributes)

        # 4. Initializing a Learner
        learner = NaiveBayes(labels, attributes_scheme['nominal'])

        actual_drift_points = [i for i in range(5000, 155000, 5000)]
        drift_acceptance_interval = 200

        # 5. Creating a Prequential Evaluation Process
        prequential = PrequentialDriftEvaluator(learner, detector, attributes, attributes_scheme,
                                                actual_drift_points, drift_acceptance_interval, project)

        result = prequential.run(stream_records, 1)
        accuracy.append(100-result[0])
        dt.append(result[1])
        fn.append(30-result[2])
        tn.append(155000-30-result[3])
        tp.append(result[2])
        fp.append(result[3])
        a = result[2]+ result[3]
        b = result[2]+(30-result[2])
        c = (155000-30-result[3])+result[3]
        d = (155000-30-result[3])+(30-result[2])
        MCC = ((result[2]*(155000-30-result[3]))-result[3]*(30-result[2]))/np.sqrt(a*b*c*d)
        mcc.append(MCC)
        mem_usage.append(result[4])
        run_time.append(result[5])
        kappa.append(result[6])

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
    m_accuracy[detector.DETECTOR_NAME] = mean_accuracy
    sd_accuracy[detector.DETECTOR_NAME] = std_accuracy
    m_kappa[detector.DETECTOR_NAME] = mean_kappa
    sd_kappa[detector.DETECTOR_NAME] = std_kappa
    m_dt[detector.DETECTOR_NAME] = mean_dt
    sd_dt[detector.DETECTOR_NAME] = std_dt
    m_fp[detector.DETECTOR_NAME] = mean_fp
    sd_fp[detector.DETECTOR_NAME] = std_fp
    m_tp[detector.DETECTOR_NAME] = mean_tp
    sd_tp[detector.DETECTOR_NAME] = std_tp

    m_fn[detector.DETECTOR_NAME] = mean_fn
    sd_fn[detector.DETECTOR_NAME] = std_fn
    m_tn[detector.DETECTOR_NAME] = mean_tn
    sd_tn[detector.DETECTOR_NAME] = std_tn
    m_mcc[detector.DETECTOR_NAME] = mean_mcc
    sd_mcc[detector.DETECTOR_NAME] = std_mcc

    m_run_time[detector.DETECTOR_NAME] = mean_run_time
    sd_run_time[detector.DETECTOR_NAME] = std_run_time
    m_mem_usage[detector.DETECTOR_NAME] = mean_mem_usage
    sd_mem_usage[detector.DETECTOR_NAME] = std_mem_usage


f = open('base_detector_circle_abrupt.txt','a')
for i,j in m_accuracy.items():
    f.write(f'\n{i}: mean_accuracy {j},\n')
for i,j in sd_accuracy.items():
    f.write(f'\n{i}: std_accuracy {j},\n')
for i,j in m_fp.items():
    f.write(f'\n{i}: mean_fp {j},\n')
for i,j in sd_fp.items():
    f.write(f'\n{i}: std_fp {j},\n')
for i,j in m_tp.items():
    f.write(f'\n{i}: mean_tp {j},\n')
for i,j in sd_tp.items():
    f.write(f'\n{i}: std_tp {j},\n')
for i,j in m_dt.items():
    f.write(f'\n{i}: mean_dt {j},\n')
for i,j in sd_dt.items():
    f.write(f'\n{i}: std_dt {j},\n')
for i,j in m_mem_usage.items():
    f.write(f'\n{i}: mean_mem {j},\n')
for i,j in sd_mem_usage.items():
    f.write(f'\n{i}: std_mem {j},\n')
for i,j in m_run_time.items():
    f.write(f'\n{i}: mean_runtime {j},\n')
for i,j in sd_run_time.items():
    f.write(f'\n{i}: std_runtime {j},\n')
for i,j in m_kappa.items():
    f.write(f'\n{i}: mean_kappa {j},\n')
for i,j in sd_kappa.items():
    f.write(f'\n{i}: std_kappa {j},\n')
for i,j in m_fn.items():
    f.write(f'\n{i}: mean_fn {j},\n')
for i,j in sd_fn.items():
    f.write(f'\n{i}: std_fn {j},\n')
for i,j in m_tn.items():
    f.write(f'\n{i}: mean_tn {j},\n')
for i,j in sd_tn.items():
    f.write(f'\n{i}: std_tn {j},\n')
for i,j in m_mcc.items():
    f.write(f'\n{i}: mean_mcc {j},\n')
for i,j in sd_mcc.items():
    f.write(f'\n{i}: std_mcc {j},\n')
f.close()