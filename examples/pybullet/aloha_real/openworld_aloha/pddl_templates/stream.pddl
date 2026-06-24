(define (stream open-world-tamp)

  (:function (PoseCost ?o ?p)
             (Pose ?o ?p))

  ;--------------------------------------------------

  (:stream test-cfree-pose-pose
    :inputs (?o1 ?p1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2) (Movable ?o1))
    :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2))

  (:stream test-cfree-pregrasp-pose
    :inputs (?a ?o1 ?p1 ?g1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Grasp ?a ?o1 ?g1) (Pose ?o2 ?p2))
    :certified (CFreePregraspPose ?a ?o1 ?p1 ?g1 ?o2 ?p2))

  (:stream test-cfree-traj-pose
    :inputs (?j ?t ?o2 ?p2)
    :domain (and (Traj ?j ?t) (Pose ?o2 ?p2))
    :certified (CFreeTrajPose ?j ?t ?o2 ?p2))

  ; ;--------------------------------------------------

  ; (:stream sample-grasp
  ;   :inputs (?a ?o)
  ;   :domain (and (Graspable ?o) (Arm ?a))
  ;   :outputs (?g)
  ;   :certified (and
  ;     (Grasp ?a ?o ?g))
  ; )

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

  ; (:stream plan-pick
  ;   :inputs (?a ?o ?p ?g )
  ;   :domain (and (Arm ?a) (Pose ?o ?p) (CanPick ?o) (Grasp ?a ?o ?g) )
  ;   :outputs (?aq ?at)
  ;   :certified (and (Pick ?a ?o ?p ?g  ?aq ?at)
  ;                   (Conf ?a ?aq) (Traj ?a ?at))
  ; )

  ;--------------------------------------------------

  (:stream plan-place
    :inputs (?a ?o ?p ?g )
    :domain (and (Arm ?a) (Pose ?o ?p) (Grasp ?a ?o ?g) )
    :outputs (?aq ?at)
    :certified (and (Place ?a ?o ?p ?g  ?aq ?at)
                    (Conf ?a ?aq) (Traj ?a ?at))
  )

  ;--------------------------------------------------

  (:stream plan-learned-pick
    :inputs (?arm ?obj ?p ?lg)
    :domain (and (Arm ?arm) (Pose ?obj ?p) (Graspable ?obj) (Grasp ?arm ?obj ?lg))
    :outputs (?aq ?at)
    :certified (and
      (Conf ?arm ?aq)
      (LearnedPick ?arm ?obj ?p ?lg ?aq ?at)
      (ImitateTraj ?arm ?at))
  )

)
