# adapted from https://github.com/FIBLAB/MoveSim

"""
This file contains the metrics for evaluating generated trajectories
"""

import scipy.stats
import numpy as np
from math import sqrt, pow
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def geodistance(x1, y1, x2, y2):
    """Computes the Euclidean distance between two points (x1, y1) and (x2, y2) in meters."""
    x1 = 500. * x1  # transform to meters
    y1 = 500. * y1
    x2 = 500. * x2
    y2 = 500. * y2
    distance = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
    return distance


class EvalUtils(object):
    """
    some commonly-used evaluation tools and functions
    """

    @staticmethod
    def filter_zero(arr):
        """
        remove zero values from an array
        :param arr: np.array, input array
        :return: np.array, output array
        """
        arr = np.array(arr)
        filtered_arr = np.array(list(filter(lambda x: x != 0., arr)))
        return filtered_arr

    @staticmethod
    def arr_to_distribution(arr, min, max, bins):
        """
        convert an array to a probability distribution
        :param arr: np.array, input array
        :param min: float, minimum of converted value
        :param max: float, maximum of converted value
        :param bins: int, number of bins between min and max
        :return: np.array, output distribution array
        """

        distribution, base = np.histogram(arr[arr<=max],bins = bins,range=(min,max))

        return distribution, base

    @staticmethod
    def norm_arr_to_distribution(arr, bins=100):
        """
        normalize an array and convert it to distribution
        :param arr: np.array, input array
        :param bins: int, number of bins in [0, 1]
        :return: np.array, np.array
        """
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = EvalUtils.filter_zero(arr)
        distribution, base = np.histogram(arr, np.arange(0, 1, 1. / bins))
        return distribution, base[:-1]

    @staticmethod
    def log_arr_to_distribution(arr, min=-30., bins=100):
        """
        calculate the logarithmic value of an array and convert it to a distribution
        :param arr: np.array, input array
        :param bins: int, number of bins between min and max
        :return: np.array,
        """
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = EvalUtils.filter_zero(arr)
        arr = np.log(arr)
        distribution, base = np.histogram(arr, np.arange(min, 0., 1./bins))
        ret_dist, ret_base = [], []
        for i in range(bins):
            if int(distribution[i]) == 0:
                continue
            else:
                ret_dist.append(distribution[i])
                ret_base.append(base[i])
        return np.array(ret_dist), np.array(ret_base)

    @staticmethod
    def get_js_divergence(p1, p2):
        """
        calculate the Jensen-Shanon Divergence of two probability distributions
        :param p1:
        :param p2:
        :return:
        """
        # normalize
        p1 = p1 / (p1.sum()+1e-14)
        p2 = p2 / (p2.sum()+1e-14)
        m = (p1 + p2) / 2
        js = (0.5 * scipy.stats.entropy(p1, m)) + (0.5 * scipy.stats.entropy(p2, m))
        return js


class IndividualEval(object):
    def __init__(self, days):
        self.days = days
        self.max_locs = 40000
        self.max_distance = 141.43
        self.bins = 100
    
    def get_topk_visits_np(self, trajs, k):
        topk_visits_loc = []
        topk_visits_freq = []
        n_trajs, seq_len = trajs.shape
        for traj in trajs:
            locs, counts = np.unique(traj, return_counts=True)
            if len(counts) > k:
                topk_idx = np.argpartition(-counts, k)[:k]
                sorted_idx = topk_idx[np.argsort(-counts[topk_idx])]
            else:
                sorted_idx = np.argsort(-counts)
            top_locs = locs[sorted_idx]
            top_freqs = counts[sorted_idx].astype(np.float32) / seq_len
            # Pad if fewer than k
            pad_size = k - len(top_locs)
            if pad_size > 0:
                top_locs = np.pad(top_locs, (0,pad_size), constant_values=-1)
                top_freqs = np.pad(top_freqs, (0, pad_size), constant_values=0)
            topk_visits_loc.append(top_locs)
            topk_visits_freq.append(top_freqs)
        topk_visits_loc = np.array(topk_visits_loc, dtype=np.int32)
        topk_visits_freq = np.array(topk_visits_freq, dtype=np.float32)
        return topk_visits_loc, topk_visits_freq
    
    def get_overall_topk_visits_freq_np(self, trajs, k):
        _, topk_visits_freq = self.get_topk_visits_np(trajs, k)
        mean_freq = np.mean(topk_visits_freq, axis=0)
        return mean_freq / np.sum(mean_freq)

    def get_geodistances(self, trajs):
        distances = []
        for traj in trajs:
            xs = (traj - 1) // 200
            ys = (traj - 1) % 200
            distances.append(geodistance(xs[:-1],ys[:-1],xs[1:],ys[1:]))
        distances = np.array(distances, dtype=float)
        return distances
    
    def get_distances_np(self, trajs):
        distances = []
        for traj in trajs:
            xs = (traj - 1) // 200
            ys = (traj - 1) % 200
            dx = xs[:-1] - xs[1:]
            dy = ys[:-1] - ys[1:]
            distances.extend(np.sqrt(dx ** 2 + dy ** 2))
        distances = np.array(distances, dtype=float)
        return distances
    
    def get_durations_np(self, trajs):
        durations = []
        b, s = trajs.shape
        for traj in trajs:
            changes = np.where(np.diff(traj) != 0)[0] + 1
            segment_lengths = np.diff(np.concatenate(([0], changes, [len(traj)])))
            durations.extend(segment_lengths)
        return np.array(durations, dtype=np.float32) / s
    
    def get_gradius_np(self, trajs):
        """
        get the std of the distances of all points away from center as `gyration radius`
        :param trajs:
        :return:
        """
        gradius = []
        for traj in trajs:
            xs = (traj - 1) // 200
            ys = (traj - 1) % 200
            xcenter, ycenter = np.mean(xs), np.mean(ys)
            rad = np.sqrt((xs - xcenter) ** 2 + (ys - ycenter) ** 2)
            gradius.append(np.mean(rad))
        gradius = np.array(gradius, dtype=float)
        return gradius
    
    def get_periodicity_np(self, trajs):
        b, s = trajs.shape
        periodicity = np.array([len(np.unique(traj)) / s for traj in trajs], dtype=np.float32)
        return periodicity

    def get_geogradius(self, trajs):
        """
        get the std of the distances of all points away from center as `gyration radius`
        :param trajs:
        :return:
        """
        gradius = []
        for traj in trajs:
            xs = np.array([(t - 1) // 200 for t in traj])
            ys = np.array([(t - 1) % 200 for t in traj])
            lng1, lat1 = np.mean(xs), np.mean(ys)
            rad = []
            for i in range(len(xs)):                   
                lng2 = xs[i]
                lat2 = ys[i]
                distance = geodistance(lng1,lat1,lng2,lat2)
                rad.append(distance)
            rad = np.mean(np.array(rad, dtype=float))
            gradius.append(rad)
        gradius = np.array(gradius, dtype=float)
        return gradius

    def get_individual_jsds(self, t1, t2, plot=False, prfx=None, pth=None):
        """
        get jsd scores of individual evaluation metrics
        :param t1: test_data
        :param t2: gene_data
        :return:
        """

        d1 = self.get_distances_np(t1)
        d2 = self.get_distances_np(t2)
        d1_dist, d1_b = EvalUtils.arr_to_distribution(
            d1, 0, self.max_distance, self.bins)
        d2_dist, d2_b = EvalUtils.arr_to_distribution(
            d2, 0, self.max_distance, self.bins)
        d_jsd = EvalUtils.get_js_divergence(d1_dist, d2_dist)

        if plot:
            self.plot_hist(d1_b, d2_b, d1_dist, d2_dist, 'generated', 'gt', 'distances', prfx, pth=pth)
        
        g1 = self.get_gradius_np(t1)
        g2 = self.get_gradius_np(t2)
        g1_dist, g1_b = EvalUtils.arr_to_distribution(
            g1, 0, self.max_distance, self.bins)
        g2_dist, g2_b = EvalUtils.arr_to_distribution(
            g2, 0, self.max_distance, self.bins)
        g_jsd = EvalUtils.get_js_divergence(g1_dist, g2_dist)

        if plot:
            self.plot_hist(g1_b, g2_b, g1_dist, g2_dist, 'generated', 'gt', 'radius', prfx, pth=pth)
        
        du1 = self.get_durations_np(t1)
        du2 = self.get_durations_np(t2)
        du1_dist, du1_b = EvalUtils.arr_to_distribution(du1, 0, 1, 48*self.days)
        du2_dist, du2_b = EvalUtils.arr_to_distribution(du2, 0, 1, 48*self.days)
        du_jsd = EvalUtils.get_js_divergence(du1_dist, du2_dist)

        if plot:
            self.plot_hist(du1_b, du2_b, du1_dist, du2_dist, 'generated', 'gt', 'durations', prfx, pth=pth)
        
        #DailyLoc
        p1 = self.get_periodicity_np(t1)
        p2 = self.get_periodicity_np(t2)
        p1_dist, p1_b = EvalUtils.arr_to_distribution(p1, 0, 1, 48*self.days)
        p2_dist, p2_b = EvalUtils.arr_to_distribution(p2, 0, 1, 48*self.days)
        p_jsd = EvalUtils.get_js_divergence(p1_dist, p2_dist)

        if plot:
            self.plot_hist(p1_b, p2_b, p1_dist, p2_dist, 'generated', 'gt', 'periodicity', prfx, pth=pth)
    
        #G-rank
        l1 =  CollectiveEval.get_visits_np(t1,self.max_locs)
        l2 =  CollectiveEval.get_visits_np(t2,self.max_locs)
        l1_dist, _ = CollectiveEval.get_topk_visits_np(l1, 1000)
        l2_dist, _ = CollectiveEval.get_topk_visits_np(l2, 1000)
        mini = min(min(l1_dist), min(l2_dist))
        maxi = max(max(l1_dist), max(l2_dist))
        l1_dist, l1_b = EvalUtils.arr_to_distribution(l1_dist,mini,maxi,20)
        l2_dist, l2_b = EvalUtils.arr_to_distribution(l2_dist,mini,maxi,20)
        l_jsd = EvalUtils.get_js_divergence(l1_dist, l2_dist)

        if plot:
            self.plot_hist(l1_b, l2_b, l1_dist, l2_dist, 'generated', 'gt', 'G-rank', prfx, pth=pth)

        #I-rank
        f1 = self.get_overall_topk_visits_freq_np(t1, 20)
        f2 = self.get_overall_topk_visits_freq_np(t2, 20)
        f1_dist, f1_b = EvalUtils.arr_to_distribution(f1,0,1,10)
        f2_dist, f2_b = EvalUtils.arr_to_distribution(f2,0,1,10)
        f_jsd = EvalUtils.get_js_divergence(f1_dist, f2_dist)

        if plot:
            self.plot_hist(f1_b, f2_b, f1_dist, f2_dist, 'generated', 'gt', 'I-rank', prfx, pth=pth)

        return d_jsd, g_jsd, du_jsd, p_jsd, l_jsd, f_jsd
    
    def plot_hist(self, bin1, bin2, hist1, hist2, label1, label2, value, prefix=None, pth=None):

        bin_centers1 = 0.5 * (bin1[1:] + bin1[:-1])
        bin_centers2 = 0.5 * (bin2[1:] + bin2[:-1])

        hist1 = hist1 / (hist1.sum()+1e-14)
        hist2 = hist2 / (hist2.sum()+1e-14)

        plt.figure()

        plt.plot(bin_centers1, hist1, label=label1, alpha=0.5, drawstyle='steps-mid', color='blue')
        plt.plot(bin_centers2, hist2, label=label2, alpha=0.5, drawstyle='steps-mid', color='orange')

        plt.xlabel(value)
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(value)
        plt.grid(True)

        if pth:
            figs_dir = pth
        elif prefix:
            figs_dir = os.path.join(os.getcwd(), "figs", prefix)
        else:
            figs_dir = os.path.join(os.getcwd(), "figs")
        os.makedirs(figs_dir, exist_ok=True)
        plot_path = os.path.join(figs_dir, value+'.png')
        plt.savefig(plot_path)

        plt.close()


class CollectiveEval(object):
    """
    collective evaluation metrics
    """
    
    @staticmethod
    def get_visits_np(trajs,max_locs):
        """
        get probability distribution of visiting all locations
        :param trajs: a numpy array of shape (n, l), where each row represents a sequence of location ids (as a trajectory of length l)
        :param max_locs: the number of possible locs within the city, default=40000
        """
        visits = np.zeros(max_locs, dtype=np.float32)
        all_points = np.concatenate(trajs).astype(np.int32)
        visits = np.bincount(all_points, minlength=max_locs)
        visits = visits / np.sum(visits)
        return visits

    @staticmethod
    def get_topk_visits_np(visits, K):
        """
        get top-k visits and the corresponding locations
        :param visits: 
        :param K:
        :return:
        """
        topk_idx = np.argpartition(-visits, K)[:K]
        topk_idx = topk_idx[np.argsort(-visits[topk_idx])]
        topk_probs = visits[topk_idx]

        return topk_probs, topk_idx.tolist()

def evaluate(test_data, gene_data):
    individualEval = IndividualEval()
    print(individualEval.get_individual_jsds(test_data,gene_data))