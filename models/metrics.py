import numpy as np
from numpy.linalg import norm
from scipy.ndimage.filters import sobel
from scipy.stats import pearsonr
import cv2
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

import models.haarPsi as hpsi_module

class Metrics:
    """Metric calculation class
    """
    
    def __init__(self, target:np.ndarray, predict:np.ndarray) -> None:
        """Initialize with target (ground truth) and predicted image (not normalized)

        Parameters
        ----------
        target : np.ndarray
            Target image
        predict : np.ndarray
            Predicted image
        """
        
        self.target = target
        self.predict = predict

        self.metrics = {'SAM': self.SAM,
                        'sCC': self.sCC,
                        'CC': self.CC,
                        # 'Q4': self.Q4,
                        'HPSI': self.HPSI,
                        'SSIM': self.SSIM,
                        'PSNR': self.PSNR,
                        'RMSE': self.RMSE,
                        'ERGAS': self.ERGAS,
                        'UIQC': self.UIQC
                        }
        
    
    def getall(self) -> dict:
        """Evaluate all metrics and return dict of them

        Returns
        -------
        dict
            Dictionary of all metrics and their values
        """
        result = dict()
        for key, function in self.metrics.items():
            result[key] = function()
        return result
    

    def HPSI(self) -> float:
        import toolbox as tbx
        target = tbx.normalize_numpy(self.target) * 255
        predict = tbx.normalize_numpy(self.predict) * 255
        result = hpsi_module.haar_psi(target, predict)
        return result[0]


    def PSNR(self) -> float:
        import toolbox as tbx
        target = tbx.normalize_numpy(self.target) * 255
        predict = tbx.normalize_numpy(self.predict) * 255
        psnr = peak_signal_noise_ratio(target, predict, data_range=predict.max() - predict.min())
        return psnr


    def SSIM(self) -> float:
        import toolbox as tbx
        target = tbx.normalize_numpy(self.target) * 255
        predict = tbx.normalize_numpy(self.predict) * 255
        ssim = structural_similarity(target, predict, data_range=predict.max() - predict.min(), channel_axis=2)
        return ssim


    def sam_old(self) -> float:
        """
        :param x_true: 高光谱图像：格式：(H, W, C)
        :param x_pred: 高光谱图像：格式：(H, W, C)
        :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
        """

        sam_rad = np.zeros(self.predict.shape[0:2])
        for x in range(self.target.shape[0]):
            for y in range(self.target.shape[1]):
                tmp_pred = self.predict[x, y]
                tmp_true = self.target[x, y]
                norm_pred=norm(tmp_pred)
                norm_true=norm(tmp_true)
                if(norm_pred!=0) and (norm_true!=0):
                    a=tmp_true*tmp_pred
                    temp=np.sum(a)/norm(tmp_pred)/norm(tmp_true)
                    if temp>1.0:
                        temp=1.0
                    sam_rad[x,y] = np.arccos(temp)
                else:
                    sam_rad[x,y]=0.0

        sam_deg = sam_rad.mean()
        return sam_deg


    def SAM(self) -> float:
        """
        :param x_true: 高光谱图像：格式：(H, W, C)
        :param x_pred: 高光谱图像：格式：(H, W, C)
        :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
        """

        dot_sum = np.sum(self.target * self.predict, axis=2)
        norm_true = norm(self.target, axis=2)
        norm_pred = norm(self.predict, axis=2)

        res = np.arccos(dot_sum/norm_pred/norm_true)
        is_nan = np.nonzero(np.isnan(res))
        for (x,y) in zip(is_nan[0], is_nan[1]):
            res[x,y]=0

        sam = np.mean(res)
        return sam


    def sCC(self) -> float:
        ps_sobel = sobel(self.predict, mode='constant')
        ms_sobel = sobel(self.target, mode='constant')
        scc = 0.0
        for i in range(self.target.shape[2]):
            a = (ps_sobel[:,:,i]).reshape(self.target.shape[0] * self.target.shape[1])
            b = (ms_sobel[:,:,i]).reshape(self.target.shape[0] * self.target.shape[1])
            #print (pearsonr(ps_sobel, ms_sobel))
            scc += pearsonr(a, b)[0]
            #scc += (np.sum(ps_sobel*ms_sobel)/np.sqrt(np.sum(ps_sobel*ps_sobel))/np.sqrt(np.sum(ms_sobel*ms_sobel)))

        return scc / self.target.shape[2]


    def CC(self) -> float:
        cc = 0.0
        for i in range(self.target.shape[2]):
            a = (self.predict[:, :, i]).reshape(self.target.shape[0] * self.target.shape[1])
            b = (self.target[:, :, i]).reshape(self.target.shape[0] * self.target.shape[1])
            # print (pearsonr(ps_sobel, ms_sobel))
            cc += pearsonr(a, b)[0]


        return cc / self.target.shape[2]


    def Q4(self) -> float:
        def conjugate(a):
            sign = -1 * np.ones(a.shape)
            sign[0,:]=1
            return a*sign
        def product(a, b):
            a = a.reshape(a.shape[0],1)
            b = b.reshape(b.shape[0],1)
            R = np.dot(a, b.transpose())
            r = np.zeros(4)
            r[0] = R[0, 0] - R[1, 1] - R[2, 2] - R[3, 3]
            r[1] = R[0, 1] + R[1, 0] + R[2, 3] - R[3, 2]
            r[2] = R[0, 2] - R[1, 3] + R[2, 0] + R[3, 1]
            r[3] = R[0, 3] + R[1, 2] - R[2, 1] + R[3, 0]
            return r
        imps = np.copy(self.predict)
        imms = np.copy(self.target)
        vec_ps = imps.reshape(imps.shape[1]*imps.shape[0], imps.shape[2])
        vec_ps = vec_ps.transpose(1,0)

        vec_ms = imms.reshape(imms.shape[1]*imms.shape[0], imms.shape[2])
        vec_ms = vec_ms.transpose(1,0)

        m1 = np.mean(vec_ps, axis=1)
        d1 = (vec_ps.transpose(1,0)-m1).transpose(1,0)
        s1 = np.mean(np.sum(d1*d1, axis=0))

        m2 = np.mean(vec_ms, axis=1)
        d2 = (vec_ms.transpose(1, 0) - m2).transpose(1, 0)
        s2 = np.mean(np.sum(d2 * d2, axis=0))


        Sc = np.zeros(vec_ms.shape)
        d2 = conjugate(d2)
        for i in range(vec_ms.shape[1]):
            Sc[:,i] = product(d1[:,i], d2[:,i])
        C = np.mean(Sc, axis=1)

        Q4 = 4 * np.sqrt(np.sum(m1*m1) * np.sum(m2*m2) * np.sum(C*C)) / (s1 + s2) / (np.sum(m1 * m1) + np.sum(m2 * m2))
        return Q4


    def RMSE(self) -> float:
        d = (self.target - self.predict)**2

        rmse = np.sqrt(np.sum(d)/(d.shape[0]*d.shape[1]))
        return rmse


    def RMSE2(self, ms, ps) -> float:
        d = (ms - ps)**2

        rmse = np.sqrt(np.sum(d)/(d.shape[0]*d.shape[1]))
        return rmse


    def ERGAS(self, ratio=0.25) -> float:
        m, n, d = self.target.shape
        summed = 0.0
        for i in range(d):
            summed += (self.RMSE2(self.target[:,:,i], self.predict[:,:,i]))**2 / np.mean(self.predict[:,:,i])**2

        ergas = 100 * ratio * np.sqrt(summed/d)
        return ergas


    def UIQC(self) -> float:
        l = self.target.shape[2]
        uiqc = 0.0
        for i in range(l):
            uiqc += self.Q(self.target[:,:,i], self.predict[:,:,i])

        return uiqc/4


    def Q(self, a, b) -> float:
        a = a.reshape(a.shape[0]*a.shape[1])
        b = b.reshape(b.shape[0]*b.shape[1])
        temp=np.cov(a,b)
        d1 =  temp[0,0]
        cov = temp[0,1]
        d2 = temp[1,1]
        m1 = np.mean(a)
        m2 = np.mean(b)
        Q = 4*cov*m1*m2/(d1+d2)/(m1**2+m2**2)

        return Q


    def D_lamda(self, ps, l_ms) -> float:
        L = ps.shape[2]
        sum = 0.0
        for i in range(L):
            for j in range(L):
                if j!=i:
                    #print(np.abs(Q(ps[:, :, i], ms[:, :, j]) - Q(l_ps[:, :, i], l_ms[:, :, j])))
                    sum += np.abs(self.Q(ps[:, :, i], ps[:, :, j]) - self.Q(l_ms[:, :, i], l_ms[:, :, j]))
        return sum/L/(L-1)


    def D_s(self, ps, l_ms, pan) -> float:
        L = ps.shape[2]
        l_pan = cv2.pyrDown(pan)
        l_pan = cv2.pyrDown(l_pan)
        sum = 0.0
        for i in range(L):
            sum += np.abs(self.Q(ps[:,:,i], pan) - self.Q(l_ms[:,:,i], l_pan))
        return sum/L

