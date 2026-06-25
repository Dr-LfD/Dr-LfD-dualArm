  (:predicates

    ; Types
    (Arm ?a)
    (Movable ?o)
    (Graspable ?o)
    (Controllable ?j)
    (Droppable ?o ?b)
    (Stackable ?o ?s)
    (Region ?s)

    (CanPick ?o)
    (CanMove ?a)

    (Pose ?o ?p)
    (InitPose ?o ?p)
    (Grasp ?a ?o ?g)
    (Conf ?j ?q)
    (RestConf ?j ?q)
    (Traj ?j ?t)

    ; Static
    (Motion ?j ?q1 ?q2 ?bt)
    (Pick ?a ?o ?p ?g  ?aq ?at)
    (Place ?a ?o ?p ?g  ?aq ?at)
    (Supported ?o ?p ?s ?sp)

    (CFreePosePose ?o1 ?p1 ?o2 ?p2)
    (CFreePregraspPose ?a ?o1 ?p1 ?g1 ?o2 ?p2)
    (CFreeTrajPose ?j ?t ?o2 ?p2)

    ; Fluent
    (AtConf ?j ?q)
    (AtPose ?o ?p)
    (AtGrasp ?a ?o ?g)
    (ArmEmpty ?a)
    (In ?o ?b)
    ; (ConfidentInPose ?o ?p)
    (HasPicked ?o)
    (DoneSkill ?sk)

    ; Derived
    (Resting ?j)
    (OtherActive ?j)
    (ArmHolding ?a ?o)
    (Holding ?o)
    (On ?o ?s)
    (Supporting ?s)

    (UnsafePose ?o ?p)
    (UnsafePregrasp ?a ?o ?p ?g)
    (UnsafeTraj ?j ?t)
