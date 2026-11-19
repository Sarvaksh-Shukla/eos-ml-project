# import numpy as np
# from inference.sampler import sample_theta
# from inference.sampler import sample_theta as _s
# from inference.sampler import sample_theta
# from inference.sampler import sample_theta
# from inference.sampler import sample_theta
# from inference.sampler import sample_theta
# from inference.sampler import sample_theta
# # imports may vary; ensure correct path
# from inference.sampler import sample_theta
# from inference.sampler import sample_theta
# from inference.sampler import sample_theta
# from inference.sampler import sample_theta
# from inference.sampler import sample_theta
# # above repeated lines are harmless placeholders if your file differs; just import sample_theta once.

# # Better: run exactly
# from inference.sampler import sample_theta, make_eos
# from physics.forward_model import ForwardModel
# from eos.units import MSUN, KM

# rng = np.random.default_rng(0)
# th = sample_theta(rng)
# print("Sampled theta:", th)
# eos = make_eos(th)

# fm = ForwardModel(eos)

# rhos = np.geomspace(6e13, 3e15, 60)
# Ms, Rs = [], []
# for rc in rhos:
#     try:
#         M, R = fm.tov.integrate_star(rc)
#         Ms.append(M / MSUN)   # convert to Msun if integrate returns grams
#         Rs.append(R / 1e5)    # convert cm->km
#     except Exception as e:
#         print("Integration error at rc=", rc, e)
#         break

# Ms = np.array(Ms)
# Rs = np.array(Rs)
# if Ms.size:
#     print("max mass [Msun] = ", Ms.max())
#     idx = Ms.argmax()
#     print("corresponding R[km] = ", Rs[idx])
#     print("Mass range:", Ms.min(), Ms.max())
# else:
#     print("No masses computed.")

from inference.dataset import synth_dataset

X, y = synth_dataset(N=10)

print("Generated:", len(X))
print("First few samples:\n", X[:5])