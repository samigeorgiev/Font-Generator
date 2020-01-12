import React from 'react';

import { NavLink } from 'react-router-dom';

import Logo from 'components/Logo';
import Toolbar from 'components/navigation/Toolbar';

import styles from './index.module.css';

const layout = props => (
    <div className={styles.Layout}>
        <header>
            <div className={styles.FGLogo}>
                <NavLink to="/" exact><Logo src="FGLogo" /></NavLink>
            </div>
            <Toolbar />
            <div className={styles.GitHubLogo}>
                <NavLink to="/" exact><Logo src="GitHubLogo" /></NavLink>
            </div>
        </header>
        {props.children}
        <footer>
            <p>&copy; 2019 - {new Date().getFullYear()} All rights reserved | Sami</p>
        </footer>
    </div>
);

export default layout;