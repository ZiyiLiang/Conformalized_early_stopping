import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

"""
The following code is copied from https://github.com/msesia/chr with slight modifications
"""

def wsc(X, y, pi, delta=0.1, M=1000, verbose=False, method = None):
    # Extract lower and upper prediction bands
    # pred_l = np.min(pred,1)
    # pred_h = np.max(pred,1)

    def wsc_v(X, y, pi, delta, v, method = method):
        n = len(y)
        # cover = (y>=pred_l)*(y<=pred_h)
        # cover = y in pi
        if method == 'union':
          def find_cover(y, pi):
            for item in pi:
              if y in item:
                return 1
            else:
              return 0
          cover = []
          for i in np.arange(n):
            cov = find_cover(y[i],pi[i])
            cover.append(cov)
        else:  
          cover = [x in y for x,y in zip(y,pi)]
        
        cover = np.array([int(x) for x in cover])
        
        z = np.dot(X,v)
        # Compute mass
        z_order = np.argsort(z)
        # pdb.set_trace()
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0-delta)*n))
        ai_best = 0
        bi_best = n-1
        # cover_min = np.mean((y >= pred_l)*(y <= pred_h))
        cover_min = np.mean(cover)
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai+int(np.round(delta*n)),n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1,n-ai+1)
            coverage[np.arange(0,bi_min-ai)]=1
            bi_star = ai+np.argmin(coverage)
            cover_star = coverage[bi_star-ai]
            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star

        # pdb.set_trace()
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]

    def sample_sphere(n, p):
        v = np.random.randn(p, n)
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X.shape[1])
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    if verbose:
        for m in tqdm(range(M)):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, pi, delta, V[m])
    else:
        for m in range(M):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, pi, delta, V[m])
        
    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star

def wsc_unbiased(X, y, pi, delta=0.1, M=1000, test_size=0.75, random_state=2020, verbose=False, method = None):
    def wsc_vab(X, y, pi, v, a, b, method = method):
        # Extract lower and upper prediction bands
        # pred_l = np.min(pred,1)
        # pred_h = np.max(pred,1)
        n = len(y)
        # cover = (y>=pred_l)*(y<=pred_h)
        if method == 'union':
          def find_cover(y, pi):
            for item in pi:
              if y in item:
                return 1
            else:
              return 0
          cover = []
          for i in np.arange(n):
            cov = find_cover(y[i],pi[i])
            cover.append(cov)
        else:  
          cover = [x in y for x,y in zip(y,pi)]

        cover = np.array([int(x) for x in cover])
        z = np.dot(X,v)
        idx = np.where((z>=a)*(z<=b))
        coverage = np.mean(cover[idx])
        return coverage

    X_train, X_test, y_train, y_test, pi_train, pi_test = train_test_split(X, y, pi, test_size=test_size,
                                                                         random_state=random_state)
    # Find adversarial parameters
    wsc_star, v_star, a_star, b_star = wsc(X_train, y_train, pi_train, delta=delta, M=M, verbose=verbose, method = method)
    # Estimate coverage
    # pdb.set_trace()
    coverage = wsc_vab(X_test, y_test, pi_test, v_star, a_star, b_star, method = method)
    return coverage