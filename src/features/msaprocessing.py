# -*- coding: utf-8 -*-

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set, MutableMapping
import string
import dataclasses
import numpy as np
import dataclasses
import re
import random

DeletionMatrix = Sequence[Sequence[int]]
FeatureDict = MutableMapping[str, np.ndarray]

@dataclasses.dataclass(frozen=True)
class Identifiers:
  species_id: str = ''

@dataclasses.dataclass(frozen=True)
class Msa:
  """Class representing a parsed MSA file."""
  sequences: Sequence[str]
  deletion_matrix: DeletionMatrix
  descriptions: Sequence[str]

  def __post_init__(self):
    if not (len(self.sequences) ==
            len(self.deletion_matrix) ==
            len(self.descriptions)):
      raise ValueError(
          'All fields for an MSA must have the same length. '
          f'Got {len(self.sequences)} sequences, '
          f'{len(self.deletion_matrix)} rows in the deletion matrix and '
          f'{len(self.descriptions)} descriptions.')


  def __len__(self):
    return len(self.sequences)

  def truncate(self, max_seqs: int):
    return Msa(sequences=self.sequences[:max_seqs],
               deletion_matrix=self.deletion_matrix[:max_seqs],
               descriptions=self.descriptions[:max_seqs])

_UNIPROT_PATTERN = re.compile(
    r"""
    ^
    # UniProtKB/TrEMBL or UniProtKB/Swiss-Prot
    (?:tr|sp)
    \|
    # A primary accession number of the UniProtKB entry.
    (?P<AccessionIdentifier>[A-Za-z0-9]{6,10})
    # Occasionally there is a _0 or _1 isoform suffix, which we ignore.
    (?:_\d)?
    \|
    # TREMBL repeats the accession ID here. Swiss-Prot has a mnemonic
    # protein ID code.
    (?:[A-Za-z0-9]+)
    _
    # A mnemonic species identification code.
    (?P<SpeciesIdentifier>([A-Za-z0-9]){1,5})
    # Small BFD uses a final value after an underscore, which we ignore.
    (?:_\d+)?
    $
    """,
    re.VERBOSE)

def _parse_sequence_identifier(msa_sequence_identifier: str) -> Identifiers:
  """Gets species from an msa sequence identifier.
  The sequence identifier has the format specified by
  _UNIPROT_TREMBL_ENTRY_NAME_PATTERN or _UNIPROT_SWISSPROT_ENTRY_NAME_PATTERN.
  An example of a sequence identifier: `tr|A0A146SKV9|A0A146SKV9_FUNHE`
  Args:
    msa_sequence_identifier: a sequence identifier.
  Returns:
    An `Identifiers` instance with species_id. These
    can be empty in the case where no identifier was found.
  """
  matches = re.search(_UNIPROT_PATTERN, msa_sequence_identifier.strip())
  if matches:
    return Identifiers(
        species_id=matches.group('SpeciesIdentifier'))
  return Identifiers()


def _extract_sequence_identifier(description: str) -> Optional[str]:
  """Extracts sequence identifier from description. Returns None if no match."""
  split_description = description.split()
  if split_description:
    return split_description[0].partition('/')[0]
  else:
    return None


def get_identifiers(description: str) -> Identifiers:
  """Computes extra MSA features from the description."""
  sequence_identifier = _extract_sequence_identifier(description)
  if sequence_identifier is None:
    return Identifiers()
  else:
    return _parse_sequence_identifier(sequence_identifier)


def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:
  """Parses FASTA string and returns list of strings with amino-acid sequences.
  Arguments:
    fasta_string: The string contents of a FASTA file.
  Returns:
    A tuple of two lists:
    * A list of sequences.
    * A list of sequence descriptions taken from the comment lines. In the
      same order as the sequences.
  """
  sequences = []
  descriptions = []
  index = -1
  for line in fasta_string.splitlines()[1:]:
    line = line.strip()
    if line.startswith('>'):
      index += 1
      descriptions.append(line[1:].split("\t")[0])  # Remove the '>' at the beginning.
      sequences.append('')
      continue
    elif not line:
      continue  # Skip blank lines.
    sequences[index] += line
  return sequences, descriptions

def parse_a3m(a3m_string: str) -> Msa:
  """Parses sequences and deletion matrix from a3m format alignment.
  Args:
    a3m_string: The string contents of a a3m file. The first sequence in the
      file should be the query sequence.
  Returns:
    A tuple of:
      * A list of sequences that have been aligned to the query. These
        might contain duplicates.
      * The deletion matrix for the alignment as a list of lists. The element
        at `deletion_matrix[i][j]` is the number of residues deleted from
        the aligned sequence i at residue position j.
      * A list of descriptions, one per sequence, from the a3m file.
  """
  sequences, descriptions = parse_fasta(a3m_string)
  deletion_matrix = []
  for msa_sequence in sequences:
    deletion_vec = []
    deletion_count = 0
    for j in msa_sequence:
      if j.islower():
        deletion_count += 1
      else:
        deletion_vec.append(deletion_count)
        deletion_count = 0
    deletion_matrix.append(deletion_vec)

  # Make the MSA matrix out of aligned (deletion-free) sequences.
  deletion_table = str.maketrans('', '', string.ascii_lowercase)
  aligned_sequences = [s.translate(deletion_table) for s in sequences]

  return Msa(sequences=aligned_sequences, deletion_matrix=deletion_matrix, descriptions=descriptions)

HHBLITS_AA_TO_ID = {
    'A': 0,
    'B': 2,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'J': 20,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'O': 20,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'U': 1,
    'V': 17,
    'W': 18,
    'X': 20,
    'Y': 19,
    'Z': 3,
    '-': 21,
}

def hamming_distance(s1, s2):
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

def maximum_diversity_sample(msas,max_num_msa=256):

    select_index= []
    distance_list = []

    for k in range(len(msas.sequences)):
        distance_list.append(hamming_distance(msas.sequences[0], msas.sequences[k]))


    for i in range(max_num_msa-1):

        potential_index = distance_list.index(max(distance_list))
        select_index.append(potential_index)

        for j in range(len(msas.sequences)):

            this_distance = hamming_distance(msas.sequences[potential_index], msas.sequences[j])
            if this_distance < distance_list[j]:
                distance_list[j] = this_distance

    random.shuffle(select_index)
    selected_index = [0] + select_index
    # print(selected_index)
    # print(len(selected_index))

    return selected_index


def get_msa_256(msas: Sequence[Msa], length=32, max_num_msa=256):
    num_msa = len(msas)
    seq_length = len(msas.sequences[0])
    sequences = []
    if num_msa >= max_num_msa:
        selected_index = maximum_diversity_sample(msas)
        msa = [msas.sequences[i] for i in selected_index]
        description = [msas.descriptions[i] for i in selected_index]
    else:
        pad_seq = "-"*len(msas.sequences[0])
        msa = msas.sequences
        description = msas.descriptions
        for i in range(num_msa,max_num_msa):
            msa.append(pad_seq)
            description.append("padding"+str(i))

    num = int(seq_length/length)+1
    msa_sample = [[] for i in range(num)]
    #print(num)
    #print(seq_length)
    for i in range(max_num_msa):
        for j in range(num):
            if j < num - 1:
                msa_sample[j].append((description[i], msa[i][j * length:(j + 1) * length]))
            else:
                msa_sample[j].append((description[i], msa[i][j * length:seq_length]))

    return msa_sample






