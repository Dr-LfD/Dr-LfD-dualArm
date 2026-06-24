  ; Derived predicates
  (:derived (Resting ?j)
    (exists (?q) (and (RestConf ?j ?q)
                      (AtConf ?j ?q))))

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