  (:stream {{NAME}}
    :inputs (?a ?o ?p ?sk) 
    :domain (and (Arm ?a) ({{ARM}} ?a) ({{OBJ}} ?o) (Pose ?o ?p) (Graspable ?o) ({{SK}} ?sk))
    :outputs (?lg)
    :certified (and
      (ImitateGrasp ?sk ?a ?o ?lg)
      (Grasp ?a ?o ?lg)
      )
  )
