"""Canonical per-feature atlas builder + enrichment for the EDEN feature viewer.

This package re-keys the viewer's structural (TED) fold layer to the canonical
homologous-superfamily unit, builds the span-enrichment sidecar, embeds the dual
(span-level + position-level ``lift``) enrichment metrics into every feature JSON,
renders the structurally-aligned gallery, assembles the #53 results page, and
deploys the standalone viewer.

Modules:
  canon             fold-granularity canonicalization (pure logic, no paths)
  rebuild           re-key the atlas TED layer to the canonical superfamily unit
  span_rates        build the span-enrichment sidecar ``span_rates.json``
  enrichment        embed span_fold/cover_rate/matched_bg_rate + lift into entries
  report            assemble the #53 results page
  structure_render  TM-align correspondence + masked-Kabsch fixed-camera render
  deploy            apply the staged corrected layer + assemble the viewer
"""
