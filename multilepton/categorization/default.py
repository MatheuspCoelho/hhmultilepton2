# coding: utf-8

"""
HH -> multi-leptons selection methods.
"""

from columnflow.categorization import Categorizer, categorizer
from columnflow.columnar_util import attach_behavior
from columnflow.util import maybe_import

from multilepton.util import IF_NANO_V12, IF_NANO_V14, IF_NANO_V15

ak = maybe_import("awkward")


# Helper to handle the taggers configuration
def get_btag_info(self: Categorizer, events: ak.Array):
    year = self.config_inst.campaign.x.year

    if year in {2024, 2025, 2026}:
        btag_tagger = "UParTAK4"
        btag_discriminator = "btagUParTAK4B"
    else:
        btag_tagger = "particleNet"
        btag_discriminator = "btagPNetB"

    wp_loose = self.config_inst.x.btag_working_points[btag_tagger]["loose"]
    wp_medium = self.config_inst.x.btag_working_points[btag_tagger]["medium"]
    wp_tight = self.config_inst.x.btag_working_points[btag_tagger]["tight"]

    btag_score = events.Jet[btag_discriminator]

    return wp_loose, wp_medium, wp_tight, btag_score


# columns read by get_global_lepton_veto, used also in the "uses" in the SR and SB categorizers
GLOBAL_VETO_COLUMNS = {
    "ElectronLoose.{pt,eta,phi,mass,charge}",
    "MuonLoose.{pt,eta,phi,mass,charge}",
}

M_4L_VETO = 140.0
M_LL_VETO = 12.0


def get_global_lepton_veto(self: Categorizer, events: ak.Array) -> ak.Array:
    """
    Both vetoes are built from the loose electron and muon collections, taus are not considered, and
    both are driven by same-flavour opposite-sign (SFOS) pairs:
      1. events contaning four leptons that split into two SFOS pairs whose four-lepton invariant
         mass is below M_4L_VETO are vetoed. all 2-pair combinations are considered, so the veto also
         reaches events with more than four loose leptons,
      2. events with any single SFOS pair below M_LL_VETO are vetoed.
    """
    ele = attach_behavior(events.ElectronLoose, "Electron")
    mu = attach_behavior(events.MuonLoose, "Muon")

    lep_p4 = ak.concatenate([ele * 1, mu * 1], axis=1)
    lep_charge = ak.concatenate([ele.charge, mu.charge], axis=1)
    lep_is_mu = ak.concatenate(
        [
            ak.zeros_like(ele.charge, dtype=bool),
            ak.ones_like(mu.charge, dtype=bool),
        ],
        axis=1,
    )

    lep_idx = ak.local_index(lep_charge, axis=1)
    is_sfos = lambda a, b: (lep_is_mu[a] == lep_is_mu[b]) & ((lep_charge[a] + lep_charge[b]) == 0)

    # veto 1: any four leptons that split into two SFOS pairs and have a low four-lepton mass. four
    # objects can be paired up in three ways, and the quadruplet counts if any of them works out
    i0, i1, i2, i3 = ak.unzip(ak.combinations(lep_idx, 4, axis=1))
    two_sfos = (
        (is_sfos(i0, i1) & is_sfos(i2, i3)) |
        (is_sfos(i0, i2) & is_sfos(i1, i3)) |
        (is_sfos(i0, i3) & is_sfos(i1, i2))
    )
    m_4l = (lep_p4[i0] + lep_p4[i1] + lep_p4[i2] + lep_p4[i3]).mass
    veto_m_4l = ak.any(two_sfos & (m_4l < M_4L_VETO), axis=1)

    # veto 2: any single SFOS pair with a low invariant mass
    j0, j1 = ak.unzip(ak.combinations(lep_idx, 2, axis=1))
    m_ll = (lep_p4[j0] + lep_p4[j1]).mass
    veto_m_ll = ak.any(is_sfos(j0, j1) & (m_ll < M_LL_VETO), axis=1)

    return ~veto_m_4l & ~veto_m_ll


@categorizer(uses={"event"})
def cat_all(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # keep all events
    return events, ak.ones_like(events.event) == 1


@categorizer(uses={"channel_id"})
def cat_etau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cetau.id


@categorizer(uses={"channel_id"})
def cat_mutau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cmutau.id


@categorizer(uses={"channel_id"})
def cat_tautau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.ctautau.id


@categorizer(uses={"channel_id"})
def cat_ee(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cee.id


@categorizer(uses={"channel_id"})
def cat_mumu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cmumu.id


@categorizer(uses={"channel_id"})
def cat_emu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cemu.id


# multilepton channels
@categorizer(uses={"channel_id"})
def cat_3e(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c3e.id


@categorizer(uses={"channel_id"})
def cat_2emu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c2emu.id


@categorizer(uses={"channel_id"})
def cat_e2mu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.ce2mu.id


@categorizer(uses={"channel_id"})
def cat_3mu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c3mu.id


@categorizer(uses={"channel_id"})
def cat_4e(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c4e.id


@categorizer(uses={"channel_id"})
def cat_3emu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c3emu.id


@categorizer(uses={"channel_id"})
def cat_2e2mu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c2e2mu.id


@categorizer(uses={"channel_id"})
def cat_e3mu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.ce3mu.id


@categorizer(uses={"channel_id"})
def cat_4mu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c4mu.id


@categorizer(uses={"channel_id"})
def cat_3etau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c3etau.id


@categorizer(uses={"channel_id"})
def cat_2emutau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c2emutau.id


@categorizer(uses={"channel_id"})
def cat_e2mutau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.ce2mutau.id


@categorizer(uses={"channel_id"})
def cat_3mutau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c3mutau.id


@categorizer(uses={"channel_id"})
def cat_2e2tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c2e2tau.id


@categorizer(uses={"channel_id"})
def cat_emu2tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cemu2tau.id


@categorizer(uses={"channel_id"})
def cat_2mu2tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c2mu2tau.id


@categorizer(uses={"channel_id"})
def cat_e3tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.ce3tau.id


@categorizer(uses={"channel_id"})
def cat_mu3tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cmu3tau.id


@categorizer(uses={"channel_id"})
def cat_4tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c4tau.id


@categorizer(uses={"channel_id"})
def cat_2eSS1tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c2eSS1tau.id


@categorizer(uses={"channel_id"})
def cat_emuSS1tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cemuSS1tau.id


@categorizer(uses={"channel_id"})
def cat_2muSS1tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c2muSS1tau.id


@categorizer(uses={"channel_id"})
def cat_2eSS(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c2eSS.id


@categorizer(uses={"channel_id"})
def cat_emuSS(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cemuSS.id


@categorizer(uses={"channel_id"})
def cat_2muSS(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.c2muSS.id


@categorizer(uses={"channel_id"})
def cat_e2tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.ce2tau.id


@categorizer(uses={"channel_id"})
def cat_mu2tau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.cmu2tau.id


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_2lSS1tauOS_SR(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c2eSS1tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cemuSS1tau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2muSS1tau.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_medium = btag_score > wp_medium
    bveto = (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 1
    OS = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SR & OS & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_2lOS1tauSS_SR(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c2eSS1tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cemuSS1tau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2muSS1tau.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_medium = btag_score > wp_medium
    bveto = (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 1
    WS = events.leptons_os == 0  # WS = Wrong Sign
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SR & WS & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_2lSS1tauOS_SB(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c2eSS1tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cemuSS1tau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2muSS1tau.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_medium = btag_score > wp_medium
    bveto = (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 0
    OS = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SR & OS & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_2lOS1tauSS_SB(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c2eSS1tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cemuSS1tau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2muSS1tau.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_medium = btag_score > wp_medium
    bveto = (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 0
    WS = events.leptons_os == 0  # WS = Wrong Sign
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SR & WS & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_2lSS_SR(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c2eSS.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cemuSS.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2muSS.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_medium = btag_score > wp_medium
    bveto = (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 1
    SS = events.leptons_os == 0
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SR & SS & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_2lOS_SR(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c2eSS.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cemuSS.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2muSS.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_medium = btag_score > wp_medium
    bveto = (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 1
    OS = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SR & OS & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_2lSS_SB(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c2eSS.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cemuSS.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2muSS.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_medium = btag_score > wp_medium
    bveto = (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 0
    SS = events.leptons_os == 0
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SR & SS & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_2lOS_SB(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c2eSS.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cemuSS.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2muSS.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_medium = btag_score > wp_medium
    bveto = (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 0
    OS = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SR & OS & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_1l2tau_SR(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.ce2tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cmu2tau.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_medium = btag_score > wp_medium
    bveto = (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 1
    OS = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SR & OS & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_1l2tau_SB(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.ce2tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cmu2tau.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_medium = btag_score > wp_medium
    bveto = (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 0
    OS = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SR & OS & global_veto)


# 3l/4l inclusive, later split into CR / SR via Z-peak
@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_3l0tau_SR(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c3e.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c3mu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2emu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.ce2mu.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_medium = btag_score > wp_medium
    bveto = (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 1
    chargeok = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SR & chargeok & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_3l0tau_SB(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c3e.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c3mu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2emu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.ce2mu.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_medium = btag_score > wp_medium
    bveto = (ak.sum(tagged_medium, axis=1) < 1)
    SB = events.tight_sel == 0
    chargeok = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SB & chargeok & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_4l_SR(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c4e.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c3emu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2e2mu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.ce3mu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c4mu.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_medium = btag_score > wp_medium
    bveto = (ak.sum(tagged_medium, axis=1) < 1)
    SR = events.tight_sel == 1
    chargeok = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SR & chargeok & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_4l_SB(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c4e.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c3emu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2e2mu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.ce3mu.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c4mu.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_medium = btag_score > wp_medium
    bveto = (ak.sum(tagged_medium, axis=1) < 1)
    SB = events.tight_sel == 0
    chargeok = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SB & chargeok & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"), IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "Tau.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_3l1tau_SR(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c3etau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2emutau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.ce2mutau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c3mutau.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_tight = btag_score > wp_tight
    bveto = (ak.sum(tagged_tight, axis=1) < 1)
    SR = events.tight_sel == 1
    chargeok = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SR & chargeok & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "Tau.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_3l1tau_SB(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c3etau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2emutau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.ce2mutau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c3mutau.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_tight = btag_score > wp_tight
    bveto = (ak.sum(tagged_tight, axis=1) < 1)
    SB = events.tight_sel == 0
    chargeok = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SB & chargeok & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "Tau.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_2l2tau_SR(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c2e2tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cemu2tau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2mu2tau.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_tight = btag_score > wp_tight
    bveto = (ak.sum(tagged_tight, axis=1) < 1)
    SR = events.tight_sel == 1
    chargeok = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SR & chargeok & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "Tau.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_2l2tau_SB(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c2e2tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cemu2tau.id)
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.c2mu2tau.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_tight = btag_score > wp_tight
    bveto = (ak.sum(tagged_tight, axis=1) < 1)
    SB = events.tight_sel == 0
    chargeok = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SB & chargeok & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "Tau.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_1l3tau_SR(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.ce3tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cmu3tau.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_tight = btag_score > wp_tight
    bveto = (ak.sum(tagged_tight, axis=1) < 1)
    SR = events.tight_sel == 1
    chargeok = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SR & chargeok & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Electron.charge", "Muon.charge", "Tau.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_1l3tau_SB(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.ce3tau.id
    catmask = catmask | (events.channel_id == self.config_inst.channels.n.cmu3tau.id)
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_tight = btag_score > wp_tight
    bveto = (ak.sum(tagged_tight, axis=1) < 1)
    SB = events.tight_sel == 0
    chargeok = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SB & chargeok & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Tau.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_4tau_SR(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c4tau.id
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_tight = btag_score > wp_tight
    bveto = (ak.sum(tagged_tight, axis=1) < 1)
    SR = events.tight_sel == 1
    chargeok = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SR & chargeok & global_veto)


@categorizer(uses={"channel_id",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}"),
    "tight_sel", "Tau.charge", "leptons_os", *GLOBAL_VETO_COLUMNS})
def cat_4tau_SB(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    catmask = events.channel_id == self.config_inst.channels.n.c4tau.id
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_tight = btag_score > wp_tight
    bveto = (ak.sum(tagged_tight, axis=1) < 1)
    SB = events.tight_sel == 0
    chargeok = events.leptons_os == 1
    global_veto = get_global_lepton_veto(self, events)
    return events, (catmask & bveto & SB & chargeok & global_veto)


# bveto
@categorizer(uses={
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}")})
def cat_bveto_on(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_tight = btag_score > wp_tight
    bveto = (ak.sum(tagged_tight, axis=1) < 1)
    return events, bveto


@categorizer(uses={"ok_bdt_eormu"})
def cat_eormu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.ok_bdt_eormu == 1


@categorizer(uses={"ok_bdt_eormu",
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}")})
def cat_eormu_bveto(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged_tight = btag_score > wp_tight
    bveto = (ak.sum(tagged_tight, axis=1) < 1)
    return events, ((events.ok_bdt_eormu == 1) & bveto)


@categorizer(uses={"tight_sel_bdt"})
def cat_tight_bdt(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # tight true
    return events, events.tight_sel_bdt == 1


@categorizer(uses={"trig_match_bdt"})
def cat_trigmatch_bdt(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # trig match
    return events, events.trig_match_bdt == 1


# Tight and trigger matching flags for the physical channels
@categorizer(uses={"tight_sel"})
def cat_tight(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # tight true
    return events, events.tight_sel == 1


@categorizer(uses={"trig_match"})
def cat_trigmatch(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # trig match
    return events, events.trig_match == 1


# QCD regions
@categorizer(uses={"leptons_os"})
def cat_os(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # oppositive sign leptons
    return events, events.leptons_os == 1


@categorizer(uses={"leptons_os"})
def cat_ss(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # same sign leptons
    return events, events.leptons_os == 0


@categorizer(uses={"tau2_isolated"})
def cat_iso(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # isolated tau2
    return events, events.tau2_isolated == 1


@categorizer(uses={"tau2_isolated"})
def cat_noniso(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # noon-isolated tau2
    return events, events.tau2_isolated == 0


# kinematic regions
@categorizer(uses={"event"})
def cat_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # fully inclusive selection
    return events, ak.ones_like(events.event) == 1


@categorizer(uses={"Jet.{pt,phi}"})
def cat_2j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # two or more jets
    return events, ak.num(events.Jet.pt, axis=1) >= 2


@categorizer(uses={
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}")})
def cat_res1b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # exactly pnet b-tags
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged = events.Jet.btagPNetB > wp_medium
    return events, ak.sum(tagged, axis=1) == 1


@categorizer(uses={
    IF_NANO_V12("Jet.btagPNetB"),
    IF_NANO_V14("Jet.btagPNetB"),
    IF_NANO_V15("Jet.{btagPNetB,btagUParTAK4B}")})
def cat_res2b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # at least two medium pnet b-tags
    wp_loose, wp_medium, wp_tight, btag_score = get_btag_info(self, events)
    tagged = events.Jet.btagPNetB > wp_medium
    return events, ak.sum(tagged, axis=1) >= 2


@categorizer(uses={cat_res1b, cat_res2b, "FatJet.{pt,phi}"})
def cat_boosted(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # not res1b or res2b, and exactly one selected fat jet that should also pass a tighter pt cut
    # note: this is just a draft
    mask = (
        (ak.num(events.FatJet, axis=1) == 1) &
        (ak.sum(events.FatJet.pt > 350, axis=1) == 1) &
        ~self[cat_res1b](events, **kwargs)[1] &
        ~self[cat_res2b](events, **kwargs)[1]
    )
    return events, mask


@categorizer(uses={"{Electron,Muon,Tau}.{pt,eta,phi,mass}"})
def cat_dy(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # e/mu driven DY region: mll > 40 and met < 30 (to supress tau decays into e/mu)
    leps = ak.concatenate([events.Electron * 1, events.Muon * 1, events.Tau * 1], axis=1)[:, :2]
    mask = (
        (leps.sum(axis=1).mass > 40) &
        (events[self.config_inst.x.met_name].pt < 30)
    )
    return events, mask


@cat_dy.init
def cat_dy_init(self: Categorizer) -> None:
    self.uses.add(f"{self.config_inst.x.met_name}.{{pt,phi}}")


@categorizer(uses={"{Electron,Muon,Tau}.{pt,eta,phi,mass}"})
def cat_tt(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # tt region: met > 30 (due to neutrino presence in leptonic w decays)
    mask = events[self.config_inst.x.met_name].pt > 30
    return events, mask


@cat_tt.init
def cat_tt_init(self: Categorizer) -> None:
    self.uses.add(f"{self.config_inst.x.met_name}.{{pt,phi}}")
