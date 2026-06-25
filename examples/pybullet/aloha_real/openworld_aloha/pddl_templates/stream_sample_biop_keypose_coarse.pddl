  (:stream {{NAME}}
    :inputs ({{INPUTS}})
    :domain (and {{DOMAIN_TERMS}})
    :outputs ({{OUTPUTS}})
    :certified (and
      (Conf ?a1 ?lc1)
      (Conf ?a2 ?lc2)
      (BiOpConf ?sk ?lc1 ?lc2)
      (GeomState ?sk ?effGeom)
      {{EXTRA_CERTIFIED_TERMS}})
  )
