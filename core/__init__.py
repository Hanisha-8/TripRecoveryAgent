"""Deterministic core — no LLM, no prompts, no judgement.

Everything in this package is a pure function of its inputs. The deep agent in
`recovery/` calls into it through thin tool wrappers, but nothing here depends on
the agent, which is why the same code can serve the agent's investigation and the
post-approval validation the agent is not trusted to perform.
"""
