# %% [markdown]
# ## Using CAVEclient to query data

# %%
from caveclient import CAVEclient
client = CAVEclient()
client.auth.get_new_token()

# %% [markdown]
# If you have successfully added your token or already have a valid token saved from a previous session, you should be able to initialize the client with the `minnie65_public` datastack:
# 
# ```python
# client = CAVEclient('minnie65_public')
# ```
# 
# If your previous saved token is invalidated, you will need to save your newly acquired token with the argument `overwrite=True`.
# 
# ```python
# client.auth.save_token(token="PASTE_YOUR_TOKEN_HERE", overwrite=True)
# ```

# %%
# client.auth.save_token(token="f721636829cadef60f6a842ebcccedc9", overwrite=True)
#%%
# this is the datastack name of the public release
# passing it will auto-configure many of the services
client = CAVEclient('minnie65_public')
client.materialize.version = 661

# %%
# View available annotation tables
client.materialize.get_tables()
# client.materialize.get_table_metadata('baylor_gnn_cell_type_fine_model_v2')
# %% [markdown]
# ### Query functionally matched EM neurons

# %%
import pandas as pd
adj_mat = pd.read_csv(f'../data/v{client.materialize.version:d}/adjacency_matrix.csv', index_col=0)
# adj_mat = pd.read_csv(f'../data/v117/adjacency_matrix.csv', index_col=0)
# %%
matched_df = client.materialize.query_table('coregistration_manual_v3')
matched_df = matched_df[matched_df.pt_root_id.isin(adj_mat.index.values)]
#%%
print(matched_df.shape)
matched_df.groupby('pt_root_id').valid.count().hist(bins=100)
#%%
import matplotlib.pyplot as plt
session_groups = matched_df.groupby(['session', 'scan_idx']).valid.count()
session_groups.name = 'num_matched_units'
session_groups.hist(bins=17, range=(0.5,17.5), width=1, align='mid', ec='navy', fc='salmon', lw=2)
plt.grid(False)
plt.xlabel('number of matched units', fontsize=20)
plt.ylabel('number of sessions', fontsize=20)
plt.xticks(range(1, 18))
plt.yticks(range(0, 3))
#%%
group_id = session_groups.copy()
group_id.name = 'group_id'
group_id[:] = np.arange(len(session_groups))
group_id
#%%
matched_m_df = matched_df.merge(session_groups, on=['session', 'scan_idx'])
matched_m_df = matched_m_df.merge(group_id, on=['session', 'scan_idx'])
matched_m_df
matched_m_df.to_pickle(f'../data/v{client.materialize.version:d}/matched_df.pkl')
# %% [markdown]
# # Access Functional data
from microns_phase3 import nda, utils
import numpy as np
import matplotlib.pyplot as plt

# %%
functional_data = {'time_axis': [], 'spike_trace': [], 'calcium_trace': [], 'pupil_radius': [], 'treadmill': []}
for i in range(len(matched_df)):
    entry = matched_df.iloc[i:i+1]
    unit_key = entry[['session', 'scan_idx', 'unit_id']].to_dict(orient='records')[0]
    nframes, fps = (nda.Scan & unit_key).fetch1('nframes', 'fps')  # fetch # frames and fps
    time_axis = np.arange(nframes)/ fps # create time axis (seconds)
    spike_trace = (nda.Activity & unit_key).fetch1('trace') # fetch spike trace
    calcium_trace = (nda.ScanUnit * nda.Fluorescence & unit_key).fetch1('trace') # fetch calcium fluorescence trace
    pupil_radius = (nda.ManualPupil & unit_key).fetch1('pupil_maj_r') # fetch manually segmented pupil trace 
    treadmill = (nda.Treadmill & unit_key).fetch1('treadmill_velocity') # fetch treadmill speed
    functional_data['time_axis'].append(time_axis)
    functional_data['spike_trace'].append(spike_trace)
    functional_data['calcium_trace'].append(calcium_trace)
    functional_data['pupil_radius'].append(pupil_radius)
    functional_data['treadmill'].append(treadmill)
# %%
# functional_data = pd.DataFrame(functional_data)
match_structure_function_df = matched_m_df.merge(functional_data, left_index=True, right_index=True)
match_structure_function_df.to_pickle(f'../data/v{client.materialize.version:d}/matched_struct_func_df.pkl')

# %% [markdown]
# ### Fetch & plot activity trace, calcium trace, pupil radius, and treadmill

# %%
nframes, fps = (nda.Scan & unit_key).fetch1('nframes', 'fps')  # fetch # frames and fps
time_axis = np.arange(nframes)/ fps # create time axis (seconds)
spike_trace = (nda.Activity & unit_key).fetch1('trace') # fetch spike trace
calcium_trace = (nda.ScanUnit * nda.Fluorescence & unit_key).fetch1('trace') # fetch calcium fluorescence trace
pupil_radius = (nda.ManualPupil & unit_key).fetch1('pupil_maj_r') # fetch manually segmented pupil trace 
treadmill = (nda.Treadmill & unit_key).fetch1('treadmill_velocity') # fetch treadmill speed

# %%
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
ax1.plot(time_axis, calcium_trace, color='g', alpha=0.3, label='calcium trace')
ax1.plot(time_axis, spike_trace, color='k', label='spike trace')
ax2.plot(time_axis, pupil_radius, color='k')
ax3.plot(time_axis, treadmill, color='k')
ax3.set_xlim(3000, 4000) 
ax1.set_ylabel('response magnitude')
ax1.legend()
ax2.set_ylabel('pupil radius')
ax3.set_ylabel('treadmill speed')
fig.suptitle(f'session: {unit_key["session"]}, scan_idx: {unit_key["scan_idx"]}, unit_id: {unit_key["unit_id"]}', fontsize=22);
[ax.spines['right'].set_visible(False) for ax in [ax1, ax2, ax3]];
[ax.spines['top'].set_visible(False) for ax in [ax1, ax2, ax3]];

# %% [markdown]
# ### Plot oracle raster

# %%
oracle_traces, score = utils.fetch_oracle_raster(unit_key)

# %%
fig,axes = plt.subplots(1,6, figsize=(6,1),dpi=300)
for col,clip_trace in zip(axes,np.moveaxis(oracle_traces,1,0)):
    col.imshow(clip_trace,cmap='binary', interpolation='nearest')
    col.set_aspect('auto')
    col.set_xticks([])
    col.set_yticks([])
axes[0].set_ylabel(f'oracle score: {score:.2f}', fontsize=5)
fig.subplots_adjust(wspace=.05)
[ax.set_title(f'oracle clip {i+1}', fontsize=6) for i, ax in enumerate(axes)];
fig.suptitle(f'session: {unit_key["session"]}, scan_idx: {unit_key["scan_idx"]}, unit_id: {unit_key["unit_id"]}', fontsize=7, y=1.2)


