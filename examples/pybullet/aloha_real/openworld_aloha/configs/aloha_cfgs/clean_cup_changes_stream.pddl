(define (stream open-world-tamp)

  (:function (PoseCost ?o ?p)
             (Pose ?o ?p))

  ;--------------------------------------------------

  (:stream test-cfree-pose-pose
    :inputs (?o1 ?p1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
    :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2))

  (:stream test-cfree-pregrasp-pose
    :inputs (?a ?o1 ?p1 ?g1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Grasp ?a ?o1 ?g1) (Pose ?o2 ?p2))
    :certified (CFreePregraspPose ?a ?o1 ?p1 ?g1 ?o2 ?p2))

  (:stream test-cfree-traj-pose
    :inputs (?j ?t ?o2 ?p2)
    :domain (and (Traj ?j ?t) (Pose ?o2 ?p2))
    :certified (CFreeTrajPose ?j ?t ?o2 ?p2))

  (:stream test-reachable
    :inputs (?a ?o ?p )
    :domain (and (Arm ?a) (Pose ?o ?p) )
    :certified (Reachable ?a ?o ?p )
  )

  ;--------------------------------------------------

  (:stream sample-placement
    :inputs (?o ?s ?sp)
    :domain (and (Stackable ?o ?s) (Pose ?s ?sp))
    :outputs (?p)
    :certified (and (Supported ?o ?p ?s ?sp) (Pose ?o ?p))
  )

  ;--------------------------------------------------

  (:stream plan-motion
    :inputs (?j ?q1 ?q2)
    :domain (and (Controllable ?j) (Conf ?j ?q1) (Conf ?j ?q2))
    :fluents (AtPose AtConf AtGrasp)
    :outputs (?t)
    :certified (Motion ?j ?q1 ?q2 ?t)
  )

  ;--------------------------------------------------

  (:stream plan-pick
    :inputs (?a ?o ?p ?g )
    :domain (and (Arm ?a) (Pose ?o ?p) (CanPick ?o) (Grasp ?a ?o ?g) )
    :outputs (?aq ?at)
    :certified (and (Pick ?a ?o ?p ?g  ?aq ?at)
                    (Conf ?a ?aq) (Traj ?a ?at))
  )

  ;--------------------------------------------------

  (:stream plan-place
    :inputs (?a ?o ?p ?g )
    :domain (and (Arm ?a) (Pose ?o ?p) (Grasp ?a ?o ?g) )
    :outputs (?aq ?at)
    :certified (and (Place ?a ?o ?p ?g  ?aq ?at)
                    (Conf ?a ?aq) (Traj ?a ?at))
  )

  ;--------------------------------------------------

  ;--------------------------------------------------

  ;--------------------------------------------------

  ;--------------------------------------------------

  ;-- per-skill instantiated streams --

  (:stream sample-grasp-traj_0
    :inputs (?a ?o ?p ?sk) 
    :domain (and (Arm ?a) (robot0 ?a) (cup ?o) (Pose ?o ?p) (Graspable ?o) (robot0_grasp_cup ?sk))
    :outputs (?lg ?aq1 ?aq2 ?at)
    :certified (and
      (ImitateGrasp ?sk ?a ?o ?lg) (Grasp ?a ?o ?lg)
      (Conf ?a ?aq1) (Conf ?a ?aq2)
      (PlanArmGripper ?a ?o ?sk ?p ?aq1 ?aq2 ?at)
      (Traj ?a ?at))
  )


  (:stream sample-grasp-traj_1
    :inputs (?a ?o ?p ?sk) 
    :domain (and (Arm ?a) (robot1 ?a) (sponge ?o) (Pose ?o ?p) (Graspable ?o) (robot1_grasp_sponge ?sk))
    :outputs (?lg ?aq1 ?aq2 ?at)
    :certified (and
      (ImitateGrasp ?sk ?a ?o ?lg) (Grasp ?a ?o ?lg)
      (Conf ?a ?aq1) (Conf ?a ?aq2)
      (PlanArmGripper ?a ?o ?sk ?p ?aq1 ?aq2 ?at)
      (Traj ?a ?at))
  )


  (:stream sample-biop-keypose_2
    :inputs (?a1 ?a2 ?sk)
    :domain (and (robot0 ?a1) (robot1 ?a2) (Skillbimanual ?sk) (bimanual_0 ?sk))
    :outputs (?lc1 ?lc2 ?effGeom)
    :certified (and
      (ImitateConf ?sk ?a1 ?lc1) (Conf ?a1 ?lc1)
      (ImitateConf ?sk ?a2 ?lc2) (Conf ?a2 ?lc2)
      (GeomState ?sk ?effGeom))
  )

  (:stream test-cfree-bioperation-pose_2
    :inputs (?a1 ?a2 ?q1 ?q2 ?o ?p ?sk)
    :domain (and (robot0 ?a1) (robot1 ?a2) (bimanual_0 ?sk) (Conf ?a1 ?q1) (Conf ?a2 ?q2) (Pose ?o ?p))
    :certified (CFreeBiOp ?a1 ?a2 ?q1 ?q2 ?o ?p ?sk)
  )


)