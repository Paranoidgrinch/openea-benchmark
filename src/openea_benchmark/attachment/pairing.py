from .model import AttachmentCandidate

def generate_attachment_candidates(neutral_branches, anion_branches):
    out = []
    for neutral in neutral_branches:
        for anion in anion_branches:
            cid = f"{neutral.branch_id}__{anion.branch_id}"
            if anion.state.charge != neutral.state.charge - 1:
                out.append(AttachmentCandidate(cid, neutral, anion, "REJECTED", ("CHARGE_CHANGE_FAILED",)))
                continue
            evidence = ["CHARGE_CHANGE_OK", "ELECTRON_ATTACHMENT_OK"]
            if neutral.state.identity_status == "CLEARED" and anion.state.identity_status == "CLEARED":
                status = "VALID"
                evidence.append("IDENTITY_CLEARED")
            else:
                status = "AMBIGUOUS"
                evidence.append("IDENTITY_UNRESOLVED")
            out.append(AttachmentCandidate(cid, neutral, anion, status, tuple(evidence)))
    return tuple(out)
