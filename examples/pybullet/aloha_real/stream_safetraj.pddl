(define (stream aloha-tamp)

  ; (:predicate (ConfConfCollision ?arm ?conf ?arm2 ?conf2)
  ;   (and (IsConf ?arm ?conf) (IsConf ?arm2 ?conf2)))

  (:stream sample-pose
    :inputs (?o ?r)
    :domain (Stackable ?o ?r)
    :outputs (?p)
    :certified (and (Pose ?o ?p) (Supported ?o ?p ?r))
  )
  (:stream sample-grasp
    :inputs (?arm ?o)
    :domain (and (IsArm ?arm) (Graspable ?o))
    :outputs (?g)
    :certified (and (Grasp ?o ?g) (IsGrasp ?arm ?o ?g))
  )
  ; certified means the predicate become true. But why fluents has 0 arity?
  (:stream inverse-kinematics
    :inputs (?arm ?o ?p ?g)
    :domain (and (IsArm ?arm) (Pose ?o ?p) (Grasp ?o ?g))
    :outputs (?q ?t)
    :certified (and (Conf ?arm ?q) (Traj ?arm ?t) (Kin ?arm ?o ?p ?g ?q ?t))
  )
  (:stream plan-free-motion
    :inputs (?arm ?q1 ?q2)
    :domain (and (IsArm ?arm) (Conf ?arm ?q1) (Conf ?arm ?q2))
    :fluents (AtPose) ; AtGrasp
    :outputs (?t)
    ;:certified (and (Traj ?t) (FreeMotion ?q1 ?t ?q2))
    :certified (FreeMotion ?arm ?q1 ?t ?q2)
  )
  (:stream plan-holding-motion
    :inputs (?arm ?q1 ?q2 ?o ?g)
    :domain (and (IsArm ?arm) (Conf ?arm ?q1) (Conf ?arm ?q2) (Grasp ?o ?g))
    :fluents (AtPose)
    :outputs (?t)
    ;:certified (and (Traj ?t) (HoldingMotion ?q1 ?t ?q2 ?o ?g))
    :certified (HoldingMotion ?arm ?q1 ?t ?q2 ?o ?g)
  )

  (:stream test-cfree-pose-pose
    :inputs (?o1 ?p1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
    :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2)
  )
  (:stream test-cfree-approach-pose
    :inputs (?o1 ?p1 ?g1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Grasp ?o1 ?g1) (Pose ?o2 ?p2))
    :certified (CFreeApproachPose ?o1 ?p1 ?g1 ?o2 ?p2)
  )
  (:stream test-cfree-traj-pose
    :inputs (?arm ?t ?o2 ?p2)
    :domain (and (Traj ?arm ?t) (Pose ?o2 ?p2))
    :certified (CFreeTrajPose ?arm ?t ?o2 ?p2)
  )

  (:stream test-cfree-traj-conf
    :inputs (?arm1 ?t ?arm2 ?q2)
    :domain (and (IsArm ?arm1) (Traj ?arm1 ?t)  (IsArm ?arm2) (Conf ?arm2 ?q2) )
    :certified (CFreeTrajConf ?arm1 ?t ?arm2 ?q2)
  )
  ;(:predicate (TrajCollision ?t ?o2 ?p2)
  ;  (and (Traj ?t) (Pose ?o2 ?p2))
  ;)

)