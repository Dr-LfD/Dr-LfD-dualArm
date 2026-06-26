(define (domain open-world-tamp)
  (:requirements :strips :equality)
  (:predicates

    ; Types
    (Arm ?a)
    (Movable ?o)
    (Graspable ?o)
    (Controllable ?j)
    (Droppable ?o ?b)
    (Stackable ?o ?s)
    (Region ?s)
    (Reachable ?a ?o ?p )

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
    (Localized ?o)
    (ConfidentInPose ?o ?p)
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

    ; LearnedAttach predicates
    (PlanArmGripper ?arm ?o ?sk ?p ?aq1 ?aq2 ?at)
    (ImitateGrasp ?sk ?a ?o ?lg)


    ; LearnedBiKeyPose / LearnedUniKeyPose predicates
    (Skillbimanual ?sk)
    (ImitateConf ?sk ?a ?lc)
    (GeomState ?sk ?sstate)

    (CFreeBiOp ?arm1 ?arm2 ?q1 ?q2 ?o ?p ?sk)
    (UnsafeBiOp ?arm1 ?arm2 ?q1 ?q2 ?sk)

    (AtKP ?j)
    (OtherNotAtKP ?j)


    ; Binding: schema name -> variable (no Grounding)
    (robot0 ?a)
    (robot1 ?a)
    (cup ?o)
    (sponge ?o)
    (table ?o)
    (robot0_grasp_cup ?sk)
    (robot1_grasp_sponge ?sk)
    (bimanual_0 ?sk)
  )

  ;--------------------------------------------------

  (:action Transit
    :parameters (?j ?q1 ?q2 ?t)
    :precondition (and (Motion ?j ?q1 ?q2 ?t)
                        (ArmEmpty ?j)
                       (CanMove ?j)
                       (or (not (OtherActive ?j)) (not (OtherNotAtKP ?j)))
                       (AtConf ?j ?q1))
    :effect (and (AtConf ?j ?q2)
                 (not (AtConf ?j ?q1))
                 (not (CanMove ?j))))

  (:action Transfer
    :parameters (?j ?q1 ?q2 ?o ?g ?t)
    :precondition (and (Motion ?j ?q1 ?q2 ?t)
                        (AtGrasp ?j ?o ?g)
                       (CanMove ?j)
                       (or (not (OtherActive ?j)) (not (OtherNotAtKP ?j)))
                       (AtConf ?j ?q1))
    :effect (and (AtConf ?j ?q2)
                 (not (AtConf ?j ?q1))
                 (not (CanMove ?j))))

  (:action pick
    :parameters (?a ?g ?o ?p ?aq ?at)
    :precondition (and (CanPick ?o)
                       (not (Supporting ?o))
                       (Pick ?a ?o ?p ?g ?aq ?at)
                       (AtPose ?o ?p)
                       (Reachable ?a ?o ?p)
                       (ArmEmpty ?a)
                       (AtConf ?a ?aq)
                       (not (UnsafePregrasp ?a ?o ?p ?g))
                       (not (UnsafeTraj ?a ?at)))
    :effect (and (AtGrasp ?a ?o ?g)
                 (CanMove ?a)
                 (ArmHolding ?a ?o)
                 (Holding ?o)
                 (HasPicked ?o)
                 (not (AtPose ?o ?p))
                 (not (ArmEmpty ?a))
                 (not (ConfidentInPose ?o ?p))))


  (:action place
    :parameters (?a ?g ?o ?p ?s ?sp ?aq ?at)
    :precondition (and (Place ?a ?o ?p ?g ?aq ?at)
                       (Supported ?o ?p ?s ?sp)
                       (Reachable ?a ?o ?p)
                       (AtGrasp ?a ?o ?g)
                       (AtPose ?s ?sp)
                       (AtConf ?a ?aq)
                       (not (UnsafePose ?o ?p))
                       (not (UnsafePregrasp ?a ?o ?p ?g))
                       (not (UnsafeTraj ?a ?at)))
    :effect (and (AtPose ?o ?p)
                 (ArmEmpty ?a)
                 (CanMove ?a)
                 (not (AtGrasp ?a ?o ?g))
                 (not (Localized ?o))
                 (not (ArmHolding ?a ?o))
                 (not (Holding ?o))))


  (:action learnedPick_0
    :parameters (?arm ?obj ?g  ?sk  ?p  ?aq1 ?aq2 ?at)
    :precondition (and (robot0 ?arm) (cup ?obj) (robot0_grasp_cup ?sk)
                      (CanPick ?obj) (AtPose ?obj ?p)
                        (not (Supporting ?obj))
                        (PlanArmGripper ?arm ?obj ?sk ?p ?aq1 ?aq2 ?at)
                        (Reachable ?arm ?obj ?p )
                        (ArmEmpty ?arm)
                        (AtConf ?arm ?aq1)
                        (ImitateGrasp ?sk ?arm ?obj ?g)
                        (not (UnsafeTraj ?arm ?at))
                  )
    :effect (and (AtGrasp ?arm ?obj ?g) (CanMove ?arm)
                  (ArmHolding ?arm ?obj) (Holding ?obj)
                  (HasPicked ?obj)
                  (AtConf ?arm ?aq2) (not (AtConf ?arm ?aq1))
                  (not (AtPose ?obj ?p)) (not (ArmEmpty ?arm))
                  (not (ConfidentInPose ?obj ?p))
                  (DoneSkill ?sk ) 
                  
                  )
  )


  (:action learnedPick_1
    :parameters (?arm ?obj ?g  ?sk  ?p  ?aq1 ?aq2 ?at)
    :precondition (and (robot1 ?arm) (sponge ?obj) (robot1_grasp_sponge ?sk)
                      (CanPick ?obj) (AtPose ?obj ?p)
                        (not (Supporting ?obj))
                        (PlanArmGripper ?arm ?obj ?sk ?p ?aq1 ?aq2 ?at)
                        (Reachable ?arm ?obj ?p )
                        (ArmEmpty ?arm)
                        (AtConf ?arm ?aq1)
                        (ImitateGrasp ?sk ?arm ?obj ?g)
                        (not (UnsafeTraj ?arm ?at))
                  )
    :effect (and (AtGrasp ?arm ?obj ?g) (CanMove ?arm)
                  (ArmHolding ?arm ?obj) (Holding ?obj)
                  (HasPicked ?obj)
                  (AtConf ?arm ?aq2) (not (AtConf ?arm ?aq1))
                  (not (AtPose ?obj ?p)) (not (ArmEmpty ?arm))
                  (not (ConfidentInPose ?obj ?p))
                  (DoneSkill ?sk ) 
                  
                  )
  )


  (:action BiOperation_2
    :parameters (?a1 ?a2 ?sk ?q1 ?q2 ?lstate ?o1 ?o2 ?g1 ?g2 ?g3 ?g4)
    :precondition (and
      (robot0 ?a1) (robot1 ?a2) (bimanual_0 ?sk)
      (ImitateConf ?sk ?a1 ?q1) (ImitateConf ?sk ?a2 ?q2)
      (AtConf ?a1 ?q1) (AtConf ?a2 ?q2)
      (GeomState ?sk ?lstate)
      (cup ?o1)
      (sponge ?o2)
      (AtGrasp ?a1 ?o1 ?g1)
      (AtGrasp ?a2 ?o2 ?g2)
      (ImitateGrasp ?sk ?a1 ?o1 ?g3)
      (ImitateGrasp ?sk ?a2 ?o2 ?g4)
      (not (UnsafeBiOp ?a1 ?a2 ?q1 ?q2 ?sk))
    )
    :effect (and
      (DoneSkill ?sk)
      (not (AtGrasp ?a1 ?o1 ?g1))
      (AtGrasp ?a1 ?o1 ?g3)
      (not (AtGrasp ?a2 ?o2 ?g2))
      (AtGrasp ?a2 ?o2 ?g4)
      (CanMove ?a1)
      (CanMove ?a2)
    )
  )

  ;--------------------------------------------------

  ; Derived predicates
  (:derived (Resting ?j)
    (exists (?q) (and (RestConf ?j ?q)
                      (AtConf ?j ?q))))
  (:derived (OtherActive ?j)
    (exists (?a) (and (Arm ?a) (not (= ?j ?a))
                      (not (Resting ?a)))))

  (:derived (AtKP ?j)
    (exists (?q ?sk) (and (ImitateConf ?sk ?j ?q)
                      (AtConf ?j ?q))))

  (:derived (OtherNotAtKP ?j)
    (exists (?a) (and (Arm ?a) (not (= ?j ?a))
                      (not (AtKP ?a)))))

  (:derived (On ?o ?s)
    (exists (?p ?sp) (and (Supported ?o ?p ?s ?sp)
                          (AtPose ?o ?p)))
  )

  (:derived (Supporting ?s)
    (exists (?p ?sp ?o) (and (Supported ?o ?p ?s ?sp)
                             (AtPose ?o ?p)))
  )

  (:derived (UnsafePose ?o1 ?p1) (and (Pose ?o1 ?p1)
    (exists (?o2 ?p2) (and (Pose ?o2 ?p2) (not (= ?o1 ?o2)) (Movable ?o2)
                           (not (CFreePosePose ?o1 ?p1 ?o2 ?p2))
                           (AtPose ?o2 ?p2)))))

  (:derived (UnsafePregrasp ?a ?o1 ?p1 ?g1) (and (Pose ?o1 ?p1) (Grasp ?a ?o1 ?g1)
    (exists (?o2 ?p2) (and (Pose ?o2 ?p2) (not (= ?o1 ?o2)) (Movable ?o2)
                           (not (CFreePregraspPose ?a ?o1 ?p1 ?g1 ?o2 ?p2))
                           (AtPose ?o2 ?p2)))))

  (:derived (UnsafeTraj ?j ?t) (and (Traj ?j ?t)
    (exists (?o2 ?p2) (and (Pose ?o2 ?p2) (Movable ?o2)
                           (not (CFreeTrajPose ?j ?t ?o2 ?p2))
                           (AtPose ?o2 ?p2)))))

  (:derived (UnsafeBiOp ?arm1 ?arm2 ?q1 ?q2 ?sk)
    (exists (?o ?p) (and (robot0 ?arm1) (robot1 ?arm2)
                        (AtConf ?arm1 ?q1) (AtConf ?arm2 ?q2)
                        (not (CFreeBiOp ?arm1 ?arm2 ?q1 ?q2 ?o ?p ?sk))
                        (AtPose ?o ?p) (Movable ?o) (CanPick ?o) (not (Holding ?o))
                    )
    )
  )


)