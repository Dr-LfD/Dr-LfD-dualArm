  (:action learnedPlace_{{ACTION_IDX}}
    :parameters {{PARAMS}}
    :precondition (and
      ({{ARM}} ?arm) ({{OBJ}} ?obj) ({{SK}} ?sk)
      (ImitatePose ?sk ?obj ?p)
      (PlanArmGripper ?arm ?s ?sk ?sp ?lg)
      (AtGrasp ?arm ?obj ?g){{REGION_PRE}})
    :effect (and (ArmEmpty ?arm) 
                 (not (AtGrasp ?arm ?obj ?g)) 
                 (not (ArmHolding ?arm ?obj)) (not (Holding ?obj))
                 (DoneSkill ?sk)
                 (AtPose ?obj ?p)))
