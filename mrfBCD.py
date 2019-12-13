import numpy as np
import sklearn.mixture as skm
import cv2

def update_data_loss(data_loss, pixels, pcoors, gmms):
    for i in range(2):
        _ds = -gmms[i].score_samples(pixels)
        for j, coors in enumerate(pcoors):
            data_loss[coors[0], coors[1], i] = _ds[j]

def update_ks(ks, im, gmms, alphas):
    for i in range(2):
        _coors = np.where(alphas == i)
        _pxs = im[_coors]
        _ks = gmms[i].predict(_pxs)
        for j, _k in enumerate(_ks):
            ks[_coors[0][j], _coors[1][j]] = _k

def update_rows(alphas, data_loss, ws_hor, ws_ver, flag, s, h, w):
    for i in range(h):
        if flag[i]:
            ds_temp = np.zeros((w, 2))
            #es_temp = np.zeros((w, 2))
            pre = np.zeros((w, 2))
            ds_temp[:, 0] += data_loss[i, :, 0]
            ds_temp[:, 1] += data_loss[i, :, 1]
            if i > 0:
                ds_temp[:, 0] += ws_ver[i - 1, :] * s[(alphas[i - 1, :], np.zeros(w).astype(np.int))]
                ds_temp[:, 1] += ws_ver[i - 1, :] * s[(alphas[i - 1, :], np.ones(w).astype(np.int))]
            if i < h - 1:
                ds_temp[:, 0] += ws_ver[i, :] * s[(alphas[i + 1, :], np.zeros(w).astype(np.int))]
                ds_temp[:, 1] += ws_ver[i, :] * s[(alphas[i + 1, :], np.ones(w).astype(np.int))]
            #es_temp = ds_temp[:]
            for j in range(1, w):
                if ds_temp[j - 1, 0] < ds_temp[j - 1, 1]:
                    ds_temp[j, 0] += ds_temp[j - 1, 0]
                    pre[j, 0] = 0
                    smth = ds_temp[j - 1, 0] + ws_hor[i, j - 1]
                    if ds_temp[j - 1, 1] < smth:
                        ds_temp[j, 1] += ds_temp[j - 1, 1]
                        pre[j, 1] = 1
                    else:
                        ds_temp[j, 1] += smth
                        pre[j, 1] = 0
                else:
                    ds_temp[j, 1] += ds_temp[j - 1, 1]
                    pre[j, 1] = 1
                    smth = ds_temp[j - 1, 1] + ws_hor[i, j - 1]
                    if ds_temp[j - 1, 0] < smth:
                        ds_temp[j, 0] += ds_temp[j - 1, 0]
                        pre[j, 0] = 0
                    else:
                        ds_temp[j, 0] += smth
                        pre[j, 0] = 1
            new_alphas = np.array([-1]*w)
            if ds_temp[-1, 0] < ds_temp[-1, 1]:
                new_alphas[-1] = 0
            else:
                new_alphas[-1] = 1
            for col in range(2, w + 1):
                new_alphas[-col] = pre[-col + 1, new_alphas[-col + 1]]
            alphas[i] = new_alphas[:]

def update_cols(alphas, data_loss, ws_hor, ws_ver, flag, s, h, w):
    for i in range(w):
        if flag[i]:
            ds_temp = np.zeros((h, 2))
            #es_temp = np.zeros((w, 2))
            pre = np.zeros((h, 2))
            ds_temp[:, 0] += data_loss[:, i, 0]
            ds_temp[:, 1] += data_loss[:, i, 1]
            if i > 0:
                ds_temp[:, 0] += ws_hor[:, i - 1] * s[(alphas[:, i - 1], np.zeros(h).astype(np.int))]
                ds_temp[:, 1] += ws_hor[:, i - 1] * s[(alphas[:, i - 1], np.ones(h).astype(np.int))]
            if i < w - 1:
                ds_temp[:, 0] += ws_hor[:, i] * s[(alphas[:, i + 1], np.zeros(h).astype(np.int))]
                ds_temp[:, 1] += ws_hor[:, i] * s[(alphas[:, i + 1], np.ones(h).astype(np.int))]
            #es_temp = ds_temp[:]
            for j in range(1, h):
                if ds_temp[j - 1, 0] < ds_temp[j - 1, 1]:
                    ds_temp[j, 0] += ds_temp[j - 1, 0]
                    pre[j, 0] = 0
                    smth = ds_temp[j - 1, 0] + ws_ver[j - 1, i]
                    if ds_temp[j - 1, 1] < smth:
                        ds_temp[j, 1] += ds_temp[j - 1, 1]
                        pre[j, 1] = 1
                    else:
                        ds_temp[j, 1] += smth
                        pre[j, 1] = 0
                else:
                    ds_temp[j, 1] += ds_temp[j - 1, 1]
                    pre[j, 1] = 1
                    smth = ds_temp[j - 1, 1] + ws_ver[j - 1, i]
                    if ds_temp[j - 1, 0] < smth:
                        ds_temp[j, 0] += ds_temp[j - 1, 0]
                        pre[j, 0] = 0
                    else:
                        ds_temp[j, 0] += smth
                        pre[j, 0] = 1
            new_alphas = np.array([-1]*h)
            if ds_temp[-1, 0] < ds_temp[-1, 1]:
                new_alphas[-1] = 0
            else:
                new_alphas[-1] = 1
            for row in range(2, h + 1):
                new_alphas[-row] = pre[-row + 1, new_alphas[-row + 1]]
            alphas[:, i] = new_alphas[:]

def energy(alphas, data_loss, ws_hor, ws_ver, s, h, w):
    de = 0.
    se = 0.
    for i in range(h):
        for j in range(w):
            de += data_loss[i, j, alphas[i, j]]
            if i < h - 1:
                se += s[alphas[i, j], alphas[i + 1, j]] * ws_ver[i, j]
            if j < w - 1:
                se += s[alphas[i, j], alphas[i, j + 1]] * ws_hor[i, j]
    return de, se

def update_gmms(gmms, im, ks, alphas, n_compo):
    pis = np.zeros((2, n_compo))
    for a in range(2):
        for k in range(n_compo):
            k_pixels = im[np.logical_and(ks == k, alphas == a)]
            pis[a, k] = k_pixels.shape[0]
            gmms[a].means_[k] = np.mean(k_pixels, axis=0)
            gmms[a].covariances_[k] = np.cov(k_pixels.T)
    pis /= np.tile(np.sum(pis, axis=-1, keepdims=True), (1, n_compo))
    gmms[0].weights_ = pis[0]
    gmms[1].weights_ = pis[1]

n_compo = 5
max_iter = 100
lmbd = 10
lmbd2 = 10
samples = ['flower', 'sponge', 'person']

for sample in samples:
    print(sample)
    im = cv2.imread('{}.png'.format(sample)).astype(np.float)
    height, width = im.shape[0], im.shape[1]
    fg_mask = cv2.imread('{}_fg.png'.format(sample))
    bg_mask = cv2.imread('{}_bg.png'.format(sample))
    fg_coors = (np.where(fg_mask > 0)[0][::3], np.where(fg_mask > 0)[1][::3])
    fg_coors_zipped = zip(fg_coors[0], fg_coors[1])
    bg_coors = (np.where(bg_mask > 0)[0][::3], np.where(bg_mask > 0)[1][::3])
    bg_coors_zipped = zip(bg_coors[0], bg_coors[1])
    fg_pixels = im[fg_coors] #np.reshape(im[fg_coors], (-1, 3))
    bg_pixels = im[bg_coors] #np.reshape(im[bg_coors], (-1, 3))
    fg_gmm = skm.GaussianMixture(n_components=n_compo, max_iter=200)
    bg_gmm = skm.GaussianMixture(n_components=n_compo, max_iter=200)
    fg_gmm.fit(fg_pixels)
    bg_gmm.fit(bg_pixels)
    ks = np.empty((height, width))
    ks_fg_prior = fg_gmm.predict(fg_pixels)
    ks_bg_prior = bg_gmm.predict(bg_pixels)
    for i, k in enumerate(ks_fg_prior):
        ks[fg_coors_zipped[i][0], fg_coors_zipped[i][1]] = k
    for i, k in enumerate(ks_bg_prior):
        ks[bg_coors_zipped[i][0], bg_coors_zipped[i][1]] = k
    alphas = np.array([[-1] * width] * height)
    alphas[fg_coors] = 0
    alphas[bg_coors] = 1

    uninit_coors = [(i, j) for i in range(height) for j in range(width) if alphas[i, j] == -1]
    uninit = np.array([im[coors[0], coors[1], :] for coors in uninit_coors])

    ks_fg = fg_gmm.predict(uninit)
    ds_fg = -fg_gmm.score_samples(uninit)
    ks_bg = bg_gmm.predict(uninit)
    ds_bg = -bg_gmm.score_samples(uninit)
    ks_fg_bg = np.array([ks_fg, ks_bg])
    fg_or_bg = [0 if ds_fg[i] < ds_bg[i] else 1 for i in range(len(uninit))]

    for i, fob in enumerate(fg_or_bg):
        alphas[uninit_coors[i][0], uninit_coors[i][1]] = fob
        ks[uninit_coors[i][0], uninit_coors[i][1]] = ks_fg_bg[fob, i]
        #dataLoss[uninit_coors[i][0], uninit_coors[i][1], 0] = ds_fg[i]
        #dataLoss[uninit_coors[i][0], uninit_coors[i][1], 1] = ds_bg[i]
    bayes_seg = (np.ones_like(alphas) - alphas) * 255
    cv2.imwrite('{}_bayes.png'.format(sample), bayes_seg.astype(np.uint8))

    ws_hor = np.zeros((height, width))
    ws_ver = np.zeros((height, width))
    temp = 0.
    count = 0
    for i in range(height):
        for j in range(width):
            if i + 1 < height:
                temp += np.sum((im[i, j] - im[i + 1, j]) ** 2)
                count += 1
            if j + 1 < width:
                temp += np.sum((im[i, j] - im[i, j + 1]) ** 2)
                count += 1
    beta = (2 * temp / count) ** -1
    ws_hor[:, :-1] = (np.exp(-beta * np.sum((im[:, 1:, :] - im[:, :-1, :]) ** 2, axis=-1)) + lmbd2) * lmbd
    ws_ver[:-1, :] = (np.exp(-beta * np.sum((im[1:, ...] - im[:-1, ...]) ** 2, axis=-1)) + lmbd2) * lmbd

    sij = np.ones((2,2)) - np.eye(2)

    update_gmms([fg_gmm, bg_gmm], im, ks, alphas, n_compo)

    data_loss = np.empty((height, width, 2))
    data_loss[fg_coors + (np.zeros(len(fg_coors_zipped)).astype(np.int),)] = 0.
    data_loss[fg_coors + (np.ones(len(fg_coors_zipped)).astype(np.int),)] = np.inf
    data_loss[bg_coors + (np.zeros(len(bg_coors_zipped)).astype(np.int),)] = np.inf
    data_loss[bg_coors + (np.ones(len(bg_coors_zipped)).astype(np.int),)] = 0.

    update_data_loss(data_loss, uninit, uninit_coors, [fg_gmm, bg_gmm])

    flag_rows = np.array([True] * height)
    flag_cols = np.array([True] * width)

    old_alphas = alphas[:]

    update_rows(alphas, data_loss, ws_hor, ws_ver, flag_rows, sij, height, width)
    update_cols(alphas, data_loss, ws_hor, ws_ver, flag_cols, sij, height, width)

    new_de, new_se = energy(alphas, data_loss, ws_hor, ws_ver, sij, height, width)
    new_energy = new_de + new_se
    print('Iteration 0 --- Energy: {} (DE={}, SE={})'.format(new_energy, new_de, new_se))

    #tolerance = 0

    for i in range(1, max_iter):
        old_energy = new_energy

        #'''
        old_flag_rows = flag_rows[:]
        old_flag_cols = flag_cols[:]
        flag_rows[:] = False
        flag_cols[:] = False
        for j in range(height):
            if old_flag_rows[j]:
                if np.any(old_alphas[j] != alphas[j]):
                    if j > 0:
                        flag_rows[j - 1] = True
                    if j < height - 1:
                        flag_rows[j + 1] = True
        for j in range(width):
            if old_flag_cols[j]:
                if np.any(old_alphas[:, j] != alphas[:, j]):
                    if j > 0:
                        flag_cols[j - 1] = True
                    if j < width - 1:
                        flag_cols[j + 1] = True
        #'''

        update_ks(ks, im, [fg_gmm, bg_gmm], alphas)
        update_gmms([fg_gmm, bg_gmm], im, ks, alphas, n_compo)
        update_data_loss(data_loss, uninit, uninit_coors, [fg_gmm, bg_gmm])

        old_alphas = alphas[:]

        update_rows(alphas, data_loss, ws_hor, ws_ver, flag_rows, sij, height, width)
        update_cols(alphas, data_loss, ws_hor, ws_ver, flag_cols, sij, height, width)

        new_de, new_se = energy(alphas, data_loss, ws_hor, ws_ver, sij, height, width)
        new_energy = new_de + new_se
        print('Iteration {} --- Energy: {} (DE={}, SE={})'.format(i, new_energy, new_de, new_se))

        if new_energy >= old_energy:
            #tolerance += 1
            #if tolerance > 3:
            break

    final_segs = (np.ones_like(alphas) - alphas) * 255
    cv2.imwrite('{}_seg.png'.format(sample), final_segs.astype(np.uint8))

    gt = cv2.imread('{}_gt.png'.format(sample)).astype(np.float)
    acc = float(np.sum(gt[..., 0]==final_segs)) / height / width
    print('acc: {}'.format(acc))
    print('\n')