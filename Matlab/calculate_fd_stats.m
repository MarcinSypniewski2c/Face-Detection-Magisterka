dataset = "E:\Marcin\Magisterka\Dane\Det\MFN";
dataset_name = "MFN";
dataset_lower = lower(dataset_name);
library = "yolo";

path_train = dataset + "/" + dataset_lower + "_cmfd_" + library + "_";
path_val = dataset + "/" + dataset_lower + "_imfd_" + library + "_";
stat_file = path_train + "statistics.csv";
conf_file = path_train + "confusion.csv";
stat_file2 = path_val + "statistics.csv";
conf_file2 = path_val + "confusion.csv";

stats = readmatrix(stat_file);
confs = readmatrix(conf_file);

stats2 = readmatrix(stat_file2);
confs2 = readmatrix(conf_file2);

tp = sum(confs(:,1));
fp = sum(confs(:,2));
fn = sum(confs(:,3));

tp2 = sum(confs2(:,1));
fp2 = sum(confs2(:,2));
fn2 = sum(confs2(:,3));


mean_iou = (mean(stats(:,1)) + mean(stats2(:,1)))/2
mean_dco = (mean(stats(:,2)) + mean(stats2(:,2)))/2
prec = (round(tp/(tp+fp)*100,2) + round(tp2/(tp2+fp2)*100,2))/2
rec = (round(tp/(tp+fn)*100,2) + round(tp2/(tp2+fn2)*100,2))/2