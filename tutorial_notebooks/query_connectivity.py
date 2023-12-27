# %%
from caveclient import CAVEclient
import nglui
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
def view3d(X):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x,y,z,s = X
    ax.scatter(x, y, z, s=s, alpha=1)
    ax.set_xlabel('X (voxels)')
    ax.set_ylabel('Y (voxels)')
    ax.set_zlabel('Z (voxels)')
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)
    return fig, ax

# %%
# this is the datastack name of the public release
# passing it will auto-configure many of the services
client = CAVEclient('minnie65_public')
client.materialize.version = 661
client.materialize.get_tables()

# %%

# %%
client.materialize.get_table_metadata('synapses_pni_2')
syn_df = client.materialize.query_table('synapses_pni_2', limit=10)
syn_df
# %%

# %%
# synapses_pni_2 is the synapse table (see the info metadata above)
# you can access the metadata of the synapse table here
client.materialize.get_table_metadata('proofreading_status_public_release')
client.materialize.query_table('proofreading_status_public_release', limit=10)

# The proofreading status of neurons that have been comprehensively proofread within this version. 
#  Axon and dendrite compartment status are marked separately under 'axon_status' and 'dendrite_status', 
#  as proofreading effort was applied differently to the different compartments in some cells.
#  There are three possible status values for each compartment: 'non' indicates no comprehensive proofreading.
#  'clean' indicates that all false merges have been removed, but all tips have not necessarily been followed.
#  'extended' indicates that the cell is both clean and all tips have been followed as far as a proofreader was able to.
#  The 'pt_position' is at a cell body or similar core position for the cell.
#  The column 'valid_id' provides the root id when the proofreading was last checked.
#  If the current root id in 'pt_root_id' is not the same as 'valid_id', there is no guarantee that the proofreading status is correct.
#  Very small false axon merges (axon fragments approximately 5 microns or less in length) were considered acceptable for clean neurites.
#  Note that this table does not list all edited cells, but only those with comprehensive effort toward the status mentioned here.
#  Table compiled by Sven Dorkenwald and Casey Schneider-Mizell, including work by many proofreaders and data maintained by Stelios Papadopoulos.

# so lets pick out the ones with extended axons
prf_df=client.materialize.query_table('proofreading_status_public_release', 
                                      filter_equal_dict={'status_axon':'extended'})
prf_df.shape

# %%
# another useful table is the nucleus_neuron_svm table
# that contains the list of all nucleus detections and those that were classified as neurons
client.materialize.get_table_metadata('nucleus_ref_neuron_svm')

# %%
# here you can see how many entries are in a table
client.materialize.get_annotation_count('nucleus_ref_neuron_svm')

# %%
# this is small enough that simply downloading the entire thing
# is reasonable
nuc_df = client.materialize.query_table('nucleus_ref_neuron_svm')
nuc_df.head()

# %%
x,y,z = np.asarray(list(nuc_df['pt_position'].values)).T
s = np.ones_like(x)*1e-2
fig, ax = view3d((x,y,z,s))

# %%
# you might notice that the 1st row had 
# an entry for pt_root_id = 0, which means that this nucleus was outside the segmented volume
# we can filter out those detections in the query_table
# using the filter options in query table
client.materialize.query_table?

# %%
x,y,z = np.asarray(list(prf_df['pt_position'].values)).T
s = np.ones_like(x)*1e-0
fig, ax = view3d((x,y,z,s))

# %%
# lets pull all the synapses from these.
# There is a practical upper bound of 200K synapses that can be queried in one go
# and it can be faster to execute many smaller queries in parallel
# but for code simplicity and to demonstrate filtering, lets do it in one query here.
syn_df = client.materialize.query_table('synapses_pni_2',
                                        filter_in_dict={'pre_pt_root_id': prf_df.pt_root_id.values,
                                                        'post_pt_root_id': prf_df.pt_root_id.values})
syn_df.shape

# %%
import pandas as pd
prf_df.to_pickle(f'../data/v{client.materialize.version:d}/prf_df.pkl')
syn_df.to_pickle(f'../data/v{client.materialize.version:d}/syn_df.pkl')


# prf_df = pd.read_pickle('../data/v{client.materialize.version:d}/prf_df.pkl')
# syn_df = pd.read_pickle('../data/v{client.materialize.version:d}/syn_df.pkl')


# %%
# post-neuron with highest degree is in the proofread set
pre_degree = syn_df.groupby('pre_pt_root_id').valid.count()
post_degree = syn_df.groupby('post_pt_root_id').valid.count()

# %%
# how many did we find?
syn_df.shape

# %%
# lets group them by pre_id to get a histogram of synapses out
post_degree.hist(bins=70)
plt.xlabel('number of output synapes')
plt.ylabel('number of cells')

# %%
n_neuron = syn_df.pre_pt_root_id.unique().shape[0]
print(n_neuron)
num_connections = syn_df.groupby(['pre_pt_root_id', 'post_pt_root_id']).valid.count().shape[0]
print(f"connection density: {num_connections/(n_neuron**2)*100:.2f} %")
# %%
# now lets reduce it to connections by grouping by pre and post
# and get a histogram of synapses per connection
syn_df.groupby(['pre_pt_root_id', 'post_pt_root_id']).valid.count().hist(bins=50)#bins=np.arange(0,10))
plt.yscale('log')
plt.xlabel('number of synpases in mapped connection', fontsize=16)
plt.ylabel('number of connections', fontsize=20)


# %% [markdown]
# # now grouping across connections, how many zero, single, and more than 1 connection do we have

# %%
# num soma is the same for each synapse in a connection, 
# so we just take the first synapse of each connection to get at this
conn_df = syn_df.groupby(['pre_pt_root_id', 'post_pt_root_id']).valid.first()
n_conn = len(conn_df)
print(n_conn)
conn_df.shape[0]/(pre_degree.shape[0]*post_degree.shape[0])

# %%
synapse_count = syn_df.groupby(['pre_pt_root_id', 'post_pt_root_id']).valid.count()
synapse_count.hist(bins=50)
plt.xlabel('number of synapses in connection', fontsize=20)
plt.ylabel('counts', fontsize=20)
plt.yscale('log')
print(f"connection density: {(synapse_count.shape[0]-n_neuron)/(n_neuron**2)*100:.2f} %")

# %%
adjacency_matrix = syn_df.pivot_table(index='pre_pt_root_id', columns='post_pt_root_id', values='valid', aggfunc='count', fill_value=0)
adjacency_matrix = adjacency_matrix.astype(int)
plt.pcolormesh(adjacency_matrix.values, vmax=1)
adjacency_matrix.to_csv(f'../data/v{client.materialize.version:d}/adjacency_matrix.csv')
plt.gca().set_aspect('equal')
plt.xlabel('post-synaptic neuron', fontsize=20)
plt.ylabel('pre-synaptic neuron', fontsize=20)


# %%
plt.hist(adjacency_matrix.values[~np.eye(n_neuron,dtype=bool)], ec='navy', fc='salmon', lw=2, bins=50)
plt.yscale('log')
plt.xlabel('number of synapses in connection', fontsize=20)
plt.ylabel('counts', fontsize=20)


# %%
adjacency_matrix = pd.read_csv(f'../data/v{client.materialize.version:d}/adjacency_matrix.csv', index_col=0).values
adj_no_self = adjacency_matrix.copy()
adj_no_self[np.eye(adj_no_self.shape[0], dtype=bool)] = 0
xs, ys = adj_no_self.nonzero()
x,y,z = np.asarray(list(prf_df['pt_position'].values)).T
s = np.ones_like(x)*10
fig, ax = view3d((x,y,z,s))
for xx, yy in zip(xs, ys):
    ax.plot([x[xx], x[yy]], [y[xx], y[yy]], [z[xx], z[yy]], c='orange', alpha=.2, lw=0.1)



