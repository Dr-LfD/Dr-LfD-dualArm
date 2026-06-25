  (:stream {{NAME}}
    :inputs (?a ?o ?s ?sp ?sk ?g)
    :domain (and
      (Arm ?a)
      ({{ARM}} ?a)
      ({{OBJ}} ?o)
      (Grasp ?a ?o ?g)
      {{SURFACE_DOMAIN}}
      ({{SK}} ?sk))
    :outputs (?lp ?lg)
    :certified (and
      (ImitatePose ?sk ?o ?lp)
      (Pose ?o ?lp)
      (PlanArmGripper ?a ?s ?sk ?sp ?lg))
  )
