  (:stream {{NAME}}
    :inputs (?o ?p ?sk)
    :domain (and ({{SK}} ?sk)
                 (Pose ?o ?p) (Movable ?o) (CanPick ?o) (SkillCheckObj ?sk ?o))
    :certified (CFreeMDF ?o ?p ?sk)
  )
