(define (stream mj-insertion)

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
    :inputs (?a ?o ?p ?bq)
    :domain (and (Arm ?a) (Pose ?o ?p) (InitConf @base ?bq))
    :certified (Reachable ?a ?o ?p ?bq)
  )

  ;--------------------------------------------------




  (:stream sample-grasp
    :inputs (?a ?o)
    :domain (and (Graspable ?o) (Arm ?a))
    :outputs (?g)
    :certified (Grasp ?a ?o ?g) ; TODO: condition on the arm?
  )

  (:stream sample-placement ; TODO: condition on the initial conf
    :inputs (?o ?s ?sp)
    :domain (and (Stackable ?o ?s) (Pose ?s ?sp)) ; TODO: (Reachable ?a ?s ?bq)
    :outputs (?p)
    :certified (and (Supported ?o ?p ?s ?sp) (Pose ?o ?p))
  )


  (:stream plan-motion
    :inputs (?j ?q1 ?q2)
    :domain (and (Controllable ?j) (Conf ?j ?q1) (Conf ?j ?q2))
    :fluents (AtPose AtConf AtGrasp)
    :outputs (?t)
    :certified (Motion ?j ?q1 ?q2 ?t)
  )


  ;--------------------------------------------------


  (:stream plan-pick ; stationary | parked | immobile | static | fixed
    :inputs (?a ?o ?p ?g ?bq)
    :domain (and (Arm ?a) (Pose ?o ?p) (CanPick ?o) (Grasp ?a ?o ?g) (InitConf @base ?bq))
    :outputs (?aq ?at)
    :certified (and (Pick ?a ?o ?p ?g ?bq ?aq ?at)
                    (Conf ?a ?aq) (Traj ?a ?at))
  )
  

  (:stream plan-place
    :inputs (?a ?o ?p ?g ?bq)
    ;:domain (and (Reachable ?a ?o ?p ?bq) (Grasp ?a ?o ?g))
    :domain (and (Arm ?a) (Pose ?o ?p) (Grasp ?a ?o ?g) (InitConf @base ?bq))
    :outputs (?aq ?at)
    :certified (and (Place ?a ?o ?p ?g ?bq ?aq ?at)
                    (Conf ?a ?aq) (Traj ?a ?at))
  )


;  integrated: given a latent ee pose z, compute the conf q of arm 
  (:stream sample-insertion-keypose
    :inputs (?arm1  ?arm2 ?o1 ?o2)
    :domain (and (left_arm ?arm1) (right_arm ?arm2) (socket ?o1) (peg ?o2))
    :outputs (?lg1 ?lg2 ?lc1 ?lc2 ?effGeom )
    :certified (and  (ImitateGrasp ?arm1 ?o1 ?lg1) (Grasp ?arm1 ?o1 ?lg1)                 
                     (ImitateGrasp ?arm2 ?o2 ?lg2)   (Grasp ?arm2 ?o2 ?lg2) 
                     (ImitateConf ?arm1 ?lc1)   (Conf ?arm1 ?lc1) 
                      (ImitateConf ?arm2 ?lc2)   (Conf ?arm2 ?lc2)
                (GeomState ?effGeom))
  )




  (:stream test-cfree-insertion-pose
    :inputs (?arm1 ?q1 ?arm2 ?q2 ?o1 ?g1 ?o2 ?g2 ?o ?p)
    :domain (and (Arm ?arm1)  (Arm ?arm2) (Conf ?arm1 ?q1) (Conf ?arm2 ?q2) (Grasp ?arm1 ?o1 ?g1) (Grasp ?arm2 ?o2 ?g2) (Pose ?o ?p))
    :certified (CFreeInsertion ?arm1 ?arm2 ?q1 ?q2 ?o1 ?g1 ?o2 ?g2 ?o ?p)
  )

)