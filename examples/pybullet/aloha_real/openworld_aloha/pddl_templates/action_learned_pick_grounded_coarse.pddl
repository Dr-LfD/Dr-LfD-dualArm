  (:action learnedPick_{{ACTION_IDX}}
    :parameters (?arm ?obj ?sk ?p ?lg)
    :precondition (and ({{ARM}} ?arm) ({{OBJ}} ?obj) ({{SK}} ?sk)
                      (CanPick ?obj) (AtPose ?obj ?p)
                      (not (Supporting ?obj)){{REACHABLE_PRE}}
                      (ArmEmpty ?arm)
                      (ImitateGrasp ?sk ?arm ?obj ?lg))
    :effect (and (AtGrasp ?arm ?obj ?lg)
                 (ArmHolding ?arm ?obj) (Holding ?obj)
                 (HasPicked ?obj)
                 (not (AtPose ?obj ?p)) (not (ArmEmpty ?arm))
                 (DoneSkill ?sk)))
