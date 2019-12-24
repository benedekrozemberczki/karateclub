"""Non-overlapping community detection."""

from karateclub.community.non_overlapping.label_propagation import LabelPropagation
from karateclub.community.non_overlapping.edmot import EdMot

"""Overlapping community detection."""

from karateclub.community.overlapping.ego_splitter import EgoNetSplitter
from karateclub.community.overlapping.danmf import DANMF
from karateclub.community.overlapping.nnsed import NNSED
from karateclub.community.overlapping.mnmf import M_NMF

"""Neighbourhood based node embedding techniques."""

from karateclub.node_embedding.neighbourhood.grarep import GraRep


"""Structural node embedding techniques."""

from karateclub.node_embedding.structural.graphwave import GraphWave



