import Layout from '@theme/Layout';
import styles from './index.module.css';

const workExperience = [
  {
    company: 'Isuzu Technical Center of America (ITCA)',
    period: 'Jan. 2021 – Present',
    roles: [
      {
        title: 'Lead Engineer, Autonomous – Advanced Engineering',
        period: 'Jun. 2024 – Present',
        bullets: [
          {
            text: 'Deployed perception components on NVIDIA TensorRT for autonomous driving applications.',
            demo: [
              { label: 'FCOS 3D Object Detection', url: 'https://youtu.be/hCQMdW7kJeU' },
              { label: 'FCOS Instance Segmentation', url: 'https://youtu.be/VIA2CVCuDEc' },
              { label: 'SORT Multi-Object Tracker', url: 'https://youtu.be/QRWK-9q1iZU' },
              { label: 'SCNN Lane Detection', url: 'https://youtu.be/tFX7SLOLkI4' },
            ],
          },
          { text: 'Conducted research on Transformer-based perception architectures.' },
          { text: 'Conducted research on manifold (matrix Lie group) methods for vehicle localization.' },
        ],
      },
      {
        title: 'Sr. Engineer, Autonomous and AI – Advanced Engineering',
        period: 'Jan. 2022 – May 2024',
        bullets: [
          {
            text: 'Led development of perception components: object detection, tracking, and visualization.',
            demo: [
              { label: 'MVXNet 3D Object Detection', url: 'https://youtu.be/LQHUm6AnVdg' },
            ],
          },
          { text: 'Developed structure-aware single-stage 3D object detection.' },
          {
            text: 'Developed LiDAR object detection and tracking.',
            demo: [
              { label: '3D LiDAR Multi-Object Tracking', url: 'https://youtu.be/hfo4PdJBrFU' },
            ],
          },
          {
            text: 'Developed multi-object tracking.',
            demo: [
              { label: 'Multi-Object Tracking', url: 'https://youtu.be/WCiK4gFPqtM' },
            ],
          },
          {
            text: 'Developed localization components: wheel odometry, GNSS/IMU processing, and Fast-LOAM.',
            demo: [
              { label: 'Visual SLAM', url: 'https://youtu.be/Z5XTmDap_Pk' },
            ],
          },
          { text: 'Acting scrum master; contributed to software architecture design and stack integration.' },
          { text: 'Provided technical mentorship and supervised interns.' },
        ],
      },
      {
        title: 'Autonomous Driving Engineer – Powertrain and Vehicle R&D',
        period: 'Jan. 2021 – Dec. 2021',
        bullets: [
          { text: 'Developed core sensor components for centralized sensor fusion (cameras and LiDARs).' },
          {
            text: 'Curb detection using 3D LiDAR point clouds.',
            demo: [
              { label: 'Curb Detection (Moriyama)', url: 'https://youtu.be/ifXtkfOTIvU' },
            ],
          },
          {
            text: 'Object detection using 3D LiDAR point clouds.',
            demo: [
              { label: 'Object Detection (Moriyama)', url: 'https://youtu.be/fAb4NLzuoVs' },
            ],
          },
          {
            text: 'Point cloud 3D map construction using Fast LOAM and OctoMap.',
            demo: [
              { label: '3D Map Construction (Moriyama)', url: 'https://youtu.be/j3xpDWPfBtw' },
            ],
          },
          { text: 'Software stack management with release tags and version control via git submodules.' },
          { text: 'Component integration and testing in simulation and on the real truck.' },
        ],
      },
    ],
  },
  {
    company: 'APTIV',
    period: 'Jul. 2018 – Jan. 2021',
    roles: [
      {
        title: 'Algorithm Engineer – Scene Perception Algorithm Team',
        period: 'Oct. 2020 – Jan. 2021',
        bullets: [
          { text: 'Developed unit test cases in vectorCAST for perception components.' },
        ],
      },
      {
        title: 'Algorithm Engineer – Fused Road Model (FRM) Team',
        period: 'Sep. 2018 – Sep. 2020',
        bullets: [
          {
            text: 'Developed fusion algorithms for object trail processing.',
            demo: [
              { label: 'Road Shape Estimation', url: 'https://youtu.be/nBhEnkIcLKs' },
              { label: 'Lane Centerline Prediction', url: 'https://youtu.be/2v-jDRaczJs' },
            ],
          },
          { text: 'Designed FRM state machine and mode manager.' },
          { text: 'Implemented error handling for vision, object fusion, and vehicle state inputs.' },
          { text: 'Developed FRM analysis pipeline and dashboard.' },
          { text: 'Coverity static analysis (AUTOSAR and MISRA C++); developed unit tests in Google Test.' },
        ],
      },
      {
        title: 'Algorithm Engineer – Autonomous Driving Behavior Team',
        period: 'Jul. 2018 – Aug. 2018',
        bullets: [
          { text: 'Developed prediction and cost function algorithm for cooperative social behavior.' },
          { text: 'Contributed to Ottomatika code migration from urban pilot to highway pilot.' },
        ],
      },
    ],
  },
];

const skills = [
  { label: 'Programming Languages', value: 'C++ (advanced), Python (advanced)' },
  { label: 'Robotics & Middleware', value: 'ROS 2 (custom packages, ament/CMake, rviz, rosbag workflows)' },
  { label: 'Model Deployment & Acceleration', value: 'TensorRT (engine build, ONNX import, C++/Python inference), CUDA' },
  { label: 'Computer Vision & 3D Perception', value: 'OpenCV, Point Cloud Library (PCL), Ceres Solver, g2o' },
  { label: 'Machine Learning Frameworks', value: 'PyTorch, ONNX ecosystem' },
  { label: 'Operating Systems', value: 'Linux/Unix (extensive experience with Ubuntu, shell and system commands)' },
  { label: 'Software Quality & Standards', value: 'AUTOSAR C++, MISRA C++, CERT C++ static analysis and compliance' },
  { label: 'Testing & Benchmarking', value: 'Google Test, pytest, Google Benchmark' },
  { label: 'High-Performance Computing', value: 'HPC clusters using SLURM' },
  { label: 'Documentation', value: 'LaTeX, TikZ' },
  { label: 'Build & Tooling', value: 'CMake, colcon, git' },
];

const publications = [
  <><strong>Zhang, Y.-C.</strong> (2025+) Error State Kalman Filter on Matrix Lie Group. <em>preprint</em>.</>,
  <>Zhang, W., Yu, W., Jia, Q., and <strong>Zhang, Y.-C.</strong> (2022) <a href="/paper/sweeping.pdf" target="_blank">Exploration and Sweeping for Autonomous Sweeper Truck</a>. <em>Isuzu Technical Journal</em> <strong>134</strong>, 42–51.</>,
  <><strong>Zhang, Y.-C.</strong> (2021) <a href="https://www.tandfonline.com/doi/full/10.1080/15472450.2021.1974858" target="_blank">Road Geometry Estimation Using Vehicle Trails: A Linear Mixed Model Approach</a>. <em>Journal of Intelligent Transportation Systems</em> <strong>27</strong>, 127–144.</>,
  <><strong>Zhang, Y.-C.</strong>, Sakhanenko, L. (2019) <a href="https://www.sciencedirect.com/science/article/pii/S0167715219301208" target="_blank">The Naive Bayes Classifier for Functional Data</a>. <em>Statistics &amp; Probability Letters</em> <strong>152</strong>, 137–146.</>,
  <>Chiou, J.-M., <strong>Zhang, Y.-C.</strong>, Chen, W.-H., and Chang, C.-W. (2014) <a href="http://www.tandfonline.com/doi/abs/10.1080/21680566.2014.892847" target="_blank">A Functional Data Approach to Missing Value Imputation and Outlier Detection for Traffic Flow Data</a>. <em>Transportmetrica B: Transport Dynamics</em> <strong>2</strong>, 106–129.</>,
  <>Fan, T.-H., Wang, Y.-F., and <strong>Zhang, Y.-C.</strong> (2014) <a href="http://www.tandfonline.com/doi/abs/10.1080/02664763.2014.894001" target="_blank">Bayesian Model Selection in Linear Mixed Effects Models with Autoregressive(p) Errors Using Mixture Priors</a>. <em>Journal of Applied Statistics</em> <strong>41</strong>, 1814–1829.</>,
];

const talks = [
  {
    title: <>Matrix Lie Theory for the Roboticist (<a href="/talk/MatrixLieRoboticist.pdf" target="_blank">slides</a>)</>,
    events: [
      <><a href="https://w2.math.ncu.edu.tw/academic/speeches/1223" target="_blank">National Central University</a>, Taiwan, February 2025</>,
      <><a href="https://stat-ds.ntu.edu.tw/News_Content_n_75466_s_250488.html" target="_blank">National Taiwan University</a>, Taiwan, February 2025</>,
    ],
  },
  {
    title: <>Road Geometry Estimation Using Vehicle Trails: A Linear Mixed Model Approach (<a href="/talk/RoadShapeEstimation.pdf" target="_blank">slides</a>)</>,
    events: [
      <><a href="https://w2.math.ncu.edu.tw/academic/speeches/1158" target="_blank">National Central University</a>, Taiwan, April 2022</>,
    ],
  },
];

const education = [
  {
    degree: 'Ph.D., Statistics and Probability',
    school: 'Michigan State University, USA',
    period: 'Aug. 2013 – Jun. 2018',
    advisor: <><em>Advisor:</em> <a href="http://www.stt.msu.edu/~luda/" target="_blank">Dr. Lyudmila Sakhanenko</a></>,
  },
  {
    degree: 'M.S., Statistics',
    school: 'National Central University, Taiwan',
    period: 'Sep. 2007 – Jun. 2009',
  },
  {
    degree: 'B.S., Mathematics',
    school: 'National Central University, Taiwan',
    period: 'Sep. 2003 – Jun. 2007',
  },
];

const teaching = [
  'Teaching Assistant & Instructor, Michigan State University, USA (2013–2018)',
];

const honors = [
  'College of Natural Science Dissertation Completion Fellowship, Summer 2018, Michigan State University',
  'College of Natural Science Dissertation Continuation Fellowship, Summer 2017, Michigan State University',
];

export default function Home() {
  return (
    <Layout title="Yi-Chen Zhang" description="Perception & Robotics Engineer — Autonomous Systems">
      <main className={styles.main}>

        {/* Profile */}
        <section className={styles.profile}>
          <img src="/img/profile.jpg" alt="Yi-Chen Zhang" className={styles.avatar} />
          <h1>Yi-Chen Zhang</h1>
          <p className={styles.subtitle}>Robotics and perception engineer specializing in autonomous systems, real-time perception pipelines, and high-performance C++ development.</p>
        </section>

        {/* About */}
        <section className={styles.section}>
          <h2>About</h2>
          <p>Hands-on Research Software Engineer with 8+ years of experience in autonomous driving perception, combining deep learning research, statistical modeling, and high-performance C++ implementation. Ph.D.-trained in Statistics with strong foundation in probabilistic modeling, Bayesian inference, and matrix Lie group methods for state estimation. Experienced in developing and evaluating perception algorithms (object detection, tracking, SLAM) and translating research prototypes into efficient, real-time automotive systems using PyTorch, CUDA, and TensorRT. Bridges research innovation and production-grade software engineering.</p>
        </section>

        {/* Work Experience */}
        <section className={styles.section}>
          <h2>Work Experience</h2>
          {workExperience.map((job) => (
            <div key={job.company} className={styles.job}>
              <div className={styles.jobHeader}>
                <strong>{job.company}</strong>
                <span>{job.period}</span>
              </div>
              {job.roles.map((role) => (
                <div key={role.title} className={styles.role}>
                  <div className={styles.roleHeader}>
                    <em>{role.title}</em>
                    <span>{role.period}</span>
                  </div>
                  <ul>
                    {role.bullets.map((b, i) => (
                      <li key={i}>
                        {b.text}
                        {b.demo && (
                          <div className={styles.demo}>
                            <span className={styles.demoLabel}>Demo: </span>
                            {b.demo.map((d, j) => (
                              <span key={j}>
                                <a href={d.url} target="_blank">{d.label}</a>
                                {j < b.demo.length - 1 && ', '}
                              </span>
                            ))}
                          </div>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          ))}
        </section>

        {/* Technical Skills */}
        <section className={styles.section}>
          <h2>Technical Skills</h2>
          <ul>
            {skills.map((s) => (
              <li key={s.label}><strong>{s.label}:</strong> {s.value}</li>
            ))}
          </ul>
        </section>

        {/* Publications */}
        <section className={styles.section}>
          <h2>Publications</h2>
          <ol>
            {publications.map((p, i) => <li key={i}>{p}</li>)}
          </ol>
        </section>

        {/* Research Talks */}
        <section className={styles.section}>
          <h2>Research Talks</h2>
          {talks.map((t, i) => (
            <div key={i} className={styles.talk}>
              <p><strong>{t.title}</strong></p>
              <ul>
                {t.events.map((e, j) => <li key={j}>{e}</li>)}
              </ul>
            </div>
          ))}
        </section>

        {/* Education */}
        <section className={styles.section}>
          <h2>Education</h2>
          {education.map((e) => (
            <div key={e.degree} className={styles.edu}>
              <div className={styles.eduHeader}>
                <strong>{e.degree}</strong>
                <span>{e.period}</span>
              </div>
              <p>{e.school}</p>
              {e.advisor && <p>{e.advisor}</p>}
            </div>
          ))}
        </section>

        {/* Teaching */}
        <section className={styles.section}>
          <h2>Teaching</h2>
          <ul>
            {teaching.map((t, i) => <li key={i}>{t}</li>)}
          </ul>
        </section>

        {/* Honors */}
        <section className={styles.section}>
          <h2>Honors</h2>
          <ul>
            {honors.map((h, i) => <li key={i}>{h}</li>)}
          </ul>
        </section>

      </main>
    </Layout>
  );
}
